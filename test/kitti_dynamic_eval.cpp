// Offline harness: runs the VOXEL dynamic rejection on a SemanticKITTI sequence without ROS,
// feeding the pose of each frame explicitly, and dumps the per-point classification so that it
// can be evaluated against the .label files on the ORIGINAL scan indices.
//
// Build: cmake -DBUILD_DYNAMIC_REJECTION_TESTS=ON ..
// Usage:
//   kitti_dynamic_eval --config <glim config dir> --seq <sequences/NN> --poses <poses/NN.txt>
//                      --out <dir> [--start 0] [--end N] [--pose_mode exact|lag|ros] [--lag 1]
//                      [--raw_max N]   (random subsample of the raw scan to N points, 0 = off)
//                      [--diag 1]      (also writes rej/NNNNNN.diag: per-point cluster-box flags)
//                      [--uniform_frame N] (bypass the GLIM preprocessor: the frame is N uniformly random
//                                       raw points beyond 1 m, as in the fido_eval_harness export)
//
// Pose modes (world = velodyne frame of frame 0, T_w_velo_i = Tr^-1 * P_i * Tr):
//   exact : the Kalman pose is set exactly to the GT pose of the current frame.
//   lag   : the Kalman pose is set exactly to the GT pose of frame i-lag.
//   ros   : emulates glim_ros without IMU before the pose prediction: one unstamped
//           PoseKalmanFilter::update() per frame with the GT pose of frame i-1.
//   ros_cv / ros_cv2 : current glim_ros without IMU: stamped update() with the GT pose of frame
//           i-1 / i-2 (odometry one / two scans behind), pose predicted at the scan stamp.
//
// Output: <out>/NNNNNN.pred = int32 n, then n x {int32 original_index, uint8 dynamic}: labels of the GLIM frame
//         (the points sent to odometry). When the rejection runs on its own sample of the raw scan
//         (rej_input_points > 0), <out>/rej/NNNNNN.pred holds the labels on those points, before the transfer.
#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include <glim/util/config.hpp>
#include <glim/util/raw_points.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>
#include <glim/dynamic_rejection/voxel_filtering.hpp>
#include <glim/dynamic_rejection/cluster_extractor.hpp>
#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

namespace fs = std::filesystem;
using namespace glim;

static Eigen::Isometry3d parse_3x4(std::istringstream& iss) {
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 4; c++) iss >> M(r, c);
  Eigen::Isometry3d T;
  T.matrix() = M;
  return T;
}

static Eigen::Isometry3d read_calib_tr(const fs::path& calib) {
  std::ifstream ifs(calib);
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind("Tr:", 0) == 0) {
      std::istringstream iss(line.substr(3));
      return parse_3x4(iss);
    }
  }
  throw std::runtime_error("Tr not found in " + calib.string());
}

static std::vector<Eigen::Isometry3d> read_poses(const fs::path& path, const Eigen::Isometry3d& Tr) {
  std::ifstream ifs(path);
  std::vector<Eigen::Isometry3d> poses;
  std::string line;
  const Eigen::Isometry3d Tr_inv = Tr.inverse();
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    const Eigen::Isometry3d P = parse_3x4(iss);
    Eigen::Isometry3d T = Tr_inv * P * Tr;
    // Re-orthonormalize (the file has limited precision).
    T.linear() = Eigen::Quaterniond(T.linear()).normalized().toRotationMatrix();
    poses.push_back(T);
  }
  return poses;
}

static std::vector<double> read_times(const fs::path& path) {
  std::ifstream ifs(path);
  std::vector<double> t;
  double v;
  while (ifs >> v) t.push_back(v);
  return t;
}

struct Key {
  uint32_t x, y, z;
  bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct KeyHash {
  size_t operator()(const Key& k) const { return (size_t(k.x) * 73856093u) ^ (size_t(k.y) * 19349663u) ^ (size_t(k.z) * 83492791u); }
};
static Key make_key(double x, double y, double z) {
  Key k;
  const float fx = static_cast<float>(x), fy = static_cast<float>(y), fz = static_cast<float>(z);
  std::memcpy(&k.x, &fx, 4);
  std::memcpy(&k.y, &fy, 4);
  std::memcpy(&k.z, &fz, 4);
  return k;
}

int main(int argc, char** argv) {
  std::string config_dir, seq_dir, poses_path, out_dir, pose_mode = "exact", log_level = "warn", dump_dir;
  int start = 0, end = -1, lag = 1, raw_max = 0, uniform_frame = 0;
  bool diag = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--config") config_dir = v;
    else if (a == "--seq") seq_dir = v;
    else if (a == "--poses") poses_path = v;
    else if (a == "--out") out_dir = v;
    else if (a == "--start") start = std::stoi(v);
    else if (a == "--end") end = std::stoi(v);
    else if (a == "--pose_mode") pose_mode = v;
    else if (a == "--lag") lag = std::stoi(v);
    else if (a == "--log") log_level = v;
    else if (a == "--raw_max") raw_max = std::stoi(v);
    else if (a == "--uniform_frame") uniform_frame = std::stoi(v);
    else if (a == "--diag") diag = v == "1";
    else if (a == "--dump_input") dump_dir = v;   // writes the rejection input of every frame (for other methods)
    else { std::cerr << "unknown arg " << a << std::endl; return 1; }
  }
  if (config_dir.empty() || seq_dir.empty() || poses_path.empty() || out_dir.empty()) {
    std::cerr << "usage: kitti_dynamic_eval --config D --seq D --poses F --out D [--start N --end N --pose_mode exact|lag|ros --lag N]" << std::endl;
    return 1;
  }
  spdlog::set_level(spdlog::level::from_str(log_level));

  GlobalConfig::instance(config_dir, true);
  fs::create_directories(fs::path(out_dir) / "rej");

  const Eigen::Isometry3d Tr = read_calib_tr(fs::path(seq_dir) / "calib.txt");
  const auto poses = read_poses(poses_path, Tr);
  const auto times = read_times(fs::path(seq_dir) / "times.txt");
  const int n_frames = static_cast<int>(std::min(poses.size(), times.size()));
  if (end < 0 || end > n_frames) end = n_frames;
  spdlog::warn("frames {}..{} of {}  pose_mode={} lag={}", start, end, n_frames, pose_mode, lag);

  auto kf = std::make_shared<PoseKalmanFilter>();
  auto wall_registry = std::make_shared<WallBBoxRegistry>();
  auto wall_filter = std::make_shared<WallFilter>(WallFilterConfig{}, wall_registry, kf);
  auto cluster_extractor = std::make_shared<DynamicClusterExtractor>(kf);
  auto rejection = std::make_shared<DynamicObjectRejectionCPU>(DynamicObjectRejectionParamsCPU(), kf);
  CloudPreprocessor preprocessor;

  const auto set_pose_exact = [&](const Eigen::Isometry3d& T) {
    // With no IMU prediction, a second update() with the same pose has zero innovation and
    // leaves the filtered pose exactly equal to T.
    kf->update(T);
    kf->update(T);
  };

  double sum_ms = 0.0;
  // Per-stage time [ms]: GLIM preprocessing, rejection input, wall filter, clustering, reject(), label transfer.
  const char* stage_names[6] = {"preprocess", "rej_input", "wall_filter", "clustering", "reject", "transfer"};
  double stage_ms[6] = {0, 0, 0, 0, 0, 0};
  std::mt19937 rng(42);
  for (int i = start; i < end; i++) {
    char name[32];
    std::snprintf(name, sizeof(name), "%06d", i);
    const fs::path bin = fs::path(seq_dir) / "velodyne" / (std::string(name) + ".bin");

    std::ifstream ifs(bin, std::ios::binary | std::ios::ate);
    const size_t bytes = ifs.tellg();
    ifs.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    ifs.read(reinterpret_cast<char*>(buf.data()), bytes);
    const size_t n_all = buf.size() / 4;
    // Optional random subsample of the raw scan (original indices are kept for the output).
    std::vector<int> keep;
    keep.reserve(n_all);
    for (size_t j = 0; j < n_all; j++) {
      const float* q = &buf[4 * j];
      if (uniform_frame > 0 && q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= 1.0f) continue;   // export min range
      keep.push_back(static_cast<int>(j));
    }
    if (raw_max > 0 && keep.size() > static_cast<size_t>(raw_max)) {
      std::shuffle(keep.begin(), keep.end(), rng);
      keep.resize(raw_max);
      std::sort(keep.begin(), keep.end());
    }
    const size_t n = keep.size();

    auto raw = std::make_shared<RawPoints>();
    raw->stamp = times[i];
    raw->points.resize(n);
    raw->intensities.resize(n);
    raw->times.assign(n, 0.0);
    std::unordered_map<Key, int, KeyHash> index;
    index.reserve(n * 2);
    for (size_t j = 0; j < n; j++) {
      const float* q = &buf[4 * keep[j]];
      raw->points[j] = Eigen::Vector4d(q[0], q[1], q[2], 1.0);
      raw->intensities[j] = q[3];
      index.emplace(make_key(q[0], q[1], q[2]), keep[j]);
    }

    // Pose for the rejection of frame i.
    if (pose_mode == "exact") {
      set_pose_exact(poses[i]);
    } else if (pose_mode == "lag") {
      set_pose_exact(poses[std::max(0, i - lag)]);
    } else if (pose_mode == "ros") {
      kf->update(poses[std::max(0, i - 1)]);
    } else if (pose_mode == "ros_cv" || pose_mode == "ros_cv2") {
      const int k = std::max(0, i - (pose_mode == "ros_cv" ? 1 : 2));
      kf->update(poses[k], times[k]);
    } else {
      std::cerr << "unknown pose_mode " << pose_mode << std::endl;
      return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto tl = t0;
    const auto lap = [&](int k) { const auto now = std::chrono::steady_clock::now(); stage_ms[k] += std::chrono::duration<double, std::milli>(now - tl).count(); tl = now; };
    PreprocessedFrame::Ptr frame;
    if (uniform_frame > 0) {
      // Same input as the fido_eval_harness export: N uniformly random raw points, raw scan kept.
      std::vector<int> sel(n);
      std::iota(sel.begin(), sel.end(), 0);
      std::shuffle(sel.begin(), sel.end(), rng);
      sel.resize(std::min<size_t>(n, uniform_frame));
      frame = std::make_shared<PreprocessedFrame>();
      frame->stamp = raw->stamp;
      frame->scan_end_time = raw->stamp + 0.1;
      frame->k_neighbors = 0;
      for (int j : sel) {
        frame->points.push_back(raw->points[j]);
        frame->intensities.push_back(raw->intensities[j]);
        frame->times.push_back(0.0);
      }
      frame->raw_points = raw;
    } else {
      frame = preprocessor.preprocess(raw);
    }
    lap(0);
    // Same steps as AsyncDynamicObjectRejection::run().
    rejection->set_frame_stamp(times[i]);
    const auto work = DynamicObjectRejectionCPU::make_rejection_frame(frame, rejection->params());
    const bool own_input = rejection->params().rej_input_points > 0 && work != frame;
    if (!dump_dir.empty()) {
      // <dump>/velodyne/NNNNNN.bin: the points the rejection works on (x y z intensity, float32);
      // <dump>/idx/NNNNNN.idx: their index in the original scan (int32), to evaluate on the same labels.
      fs::create_directories(fs::path(dump_dir) / "velodyne"); fs::create_directories(fs::path(dump_dir) / "idx");
      std::ofstream pv(fs::path(dump_dir) / "velodyne" / (std::string(name) + ".bin"), std::ios::binary);
      std::ofstream pi(fs::path(dump_dir) / "idx" / (std::string(name) + ".idx"), std::ios::binary);
      for (size_t k = 0; k < work->points.size(); ++k) {
        const auto& p = work->points[k];
        const auto it = index.find(make_key(p.x(), p.y(), p.z()));
        const int32_t id = it == index.end() ? -1 : it->second;
        const float v[4] = {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()),
                            k < work->intensities.size() ? static_cast<float>(work->intensities[k]) : 0.f};
        pv.write(reinterpret_cast<const char*>(v), sizeof(v));
        pi.write(reinterpret_cast<const char*>(&id), 4);
      }
    }
    lap(1);
    const WallFilterResult wf = wall_filter->filter(*work);
    lap(2);
    std::vector<BoundingBox> cluster_bboxes, historical_bboxes;
    if (wf.voxelmap) {
      cluster_bboxes = cluster_extractor->extract_clusters(wf.voxelmap, frame->stamp);
      historical_bboxes = cluster_extractor->get_dynamic_track_history();
    }
    lap(3);
    DynamicRejectionResult dr = rejection->reject(wf, own_input ? work : frame, cluster_bboxes, historical_bboxes);
    const DynamicRejectionResult dr_rej = dr;
    lap(4);
    if (own_input) DynamicObjectRejectionCPU::transfer_labels(frame, dr, rejection->params());
    cluster_extractor->update_dynamic_feedback(cluster_bboxes);
    lap(5);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sum_ms += ms;

    int unmatched = 0;
    const auto write_pred = [&](const fs::path& path, const DynamicRejectionResult& res) {
      std::vector<std::pair<int32_t, uint8_t>> out;
      const auto dump = [&](const PreprocessedFrame::Ptr& f, uint8_t label) {
        if (!f) return;
        for (const auto& p : f->points) {
          auto it = index.find(make_key(p.x(), p.y(), p.z()));
          if (it == index.end()) { unmatched++; continue; }
          out.emplace_back(it->second, label);
        }
      };
      dump(res.static_frame, 0);
      dump(res.dynamic_frame, 1);
      std::ofstream ofs(path, std::ios::binary);
      const int32_t cnt = static_cast<int32_t>(out.size());
      ofs.write(reinterpret_cast<const char*>(&cnt), 4);
      for (const auto& [idx, lab] : out) {
        ofs.write(reinterpret_cast<const char*>(&idx), 4);
        ofs.write(reinterpret_cast<const char*>(&lab), 1);
      }
      return out.size();
    };
    const size_t n_out = write_pred(fs::path(out_dir) / (std::string(name) + ".pred"), dr);
    if (own_input) write_pred(fs::path(out_dir) / "rej" / (std::string(name) + ".pred"), dr_rej);
    if (diag && own_input) {
      // Per point of the rejection input: bit0 labelled dynamic, bit1 inside a cluster box,
      // bit2 inside a box judged dynamic in this frame (before the temporal confirmation).
      std::ofstream ofs(fs::path(out_dir) / "rej" / (std::string(name) + ".diag"), std::ios::binary);
      std::vector<std::pair<int32_t, uint8_t>> rows;
      const auto add = [&](const PreprocessedFrame::Ptr& f, uint8_t lab) {
        if (!f) return;
        for (const auto& p : f->points) {
          auto it = index.find(make_key(p.x(), p.y(), p.z()));
          if (it == index.end()) continue;
          uint8_t flags = lab;
          for (const auto& b : cluster_bboxes) {
            if (b.contains(p)) { flags |= 2; if (b.is_dynamic_bbox()) flags |= 4; }
          }
          rows.emplace_back(it->second, flags);
        }
      };
      add(dr_rej.static_frame, 0);
      add(dr_rej.dynamic_frame, 1);
      const int32_t cnt = static_cast<int32_t>(rows.size());
      ofs.write(reinterpret_cast<const char*>(&cnt), 4);
      for (const auto& [idx, fl] : rows) {
        ofs.write(reinterpret_cast<const char*>(&idx), 4);
        ofs.write(reinterpret_cast<const char*>(&fl), 1);
      }
    }

    if (unmatched) spdlog::warn("frame {}: {} output points not matched to the original scan", i, unmatched);
    if ((i - start) % 100 == 0) {
      const size_t nd = dr.dynamic_frame ? dr.dynamic_frame->points.size() : 0;
      std::printf("[%d/%d] %zu raw -> %zu classified (%zu dynamic)  %.1f ms (avg %.1f)\n", i, end, n, n_out, nd, ms,
                  sum_ms / (i - start + 1));
      std::fflush(stdout);
    }
  }
  std::printf("done: %d frames, avg %.1f ms/frame\n", end - start, sum_ms / std::max(1, end - start));
  std::printf("stages avg ms:");
  for (int k = 0; k < 6; ++k) std::printf(" %s %.1f", stage_names[k], stage_ms[k] / std::max(1, end - start));
  std::printf("\n");
  return 0;
}
