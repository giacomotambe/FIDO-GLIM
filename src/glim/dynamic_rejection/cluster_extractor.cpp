/*
    FILE: cluster_extractor.cpp
    ------------------
    Dynamic cluster extractor: DBSCAN → NMS → peer merge → overlap tracking.

    Pipeline:
      1. cluster_voxels()       — DBSCAN on non-wall voxel centroids
      2. build_point_clusters() — collect raw points per cluster
      3. compute_bounding_boxes() + nms3D() — AABB per cluster, remove duplicates
      4. merge_nearby_clusters() — iteratively merge bboxes closer than peer_merge_distance
      5. update_tracks()        — overlap-based tracking with EMA velocity estimation
*/

#include <glim/dynamic_rejection/cluster_extractor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <gtsam_points/ann/kdtree.hpp>
#include <glim/util/config.hpp>

namespace glim {

// ===========================================================================
// DynamicClusterExtractorParams
// ===========================================================================

DynamicClusterExtractorParams::DynamicClusterExtractorParams() {
    spdlog::debug("[cluster_extractor] loading config");

    Config config(GlobalConfig::get_config_path("config_dynamic_cluster_extractor"));
    Config ros_config(GlobalConfig::get_config_path("config_ros"));
    eps_voxel_factor    = config.param<double>("dynamic_cluster_extractor", "eps_voxel_factor",    2.0);
    min_pts             = config.param<int>   ("dynamic_cluster_extractor", "min_pts",             1);
    knn_max_neighbors   = config.param<int>   ("dynamic_cluster_extractor", "knn_max_neighbors",   64);
    min_cluster_voxels  = config.param<int>   ("dynamic_cluster_extractor", "min_cluster_voxels",  2);
    min_points_for_bbox = config.param<int>   ("dynamic_cluster_extractor", "min_points_for_bbox", 20);

    bbox_min_extent = config.param<double>("dynamic_cluster_extractor", "bbox_min_extent", 0.0);
    bbox_max_extent = config.param<double>("dynamic_cluster_extractor", "bbox_max_extent", 1e9);
    bbox_min_volume = config.param<double>("dynamic_cluster_extractor", "bbox_min_volume", 0.0);
    bbox_max_volume = config.param<double>("dynamic_cluster_extractor", "bbox_max_volume", 1e9);

    cluster_iou_threshold = config.param<double>("dynamic_cluster_extractor", "cluster_iou_threshold", 0.5);
    env_type = ros_config.param<std::string>("glim_ros", "env_type", "OUTDOOR");
    spdlog::info("[cluster_extractor] environment type from ROS config: {}", env_type);
    if (env_type == "OUTDOOR") {
       
        peer_merge_distance = config.param<double>("dynamic_cluster_extractor", "peer_merge_distance_outdoor", 2.0);
        spdlog::info("[cluster_extractor] peer merge distance (outdoor): {}", peer_merge_distance);
    } else {
        spdlog::info("[cluster_extractor] environment type: INDOOR");
        peer_merge_distance = config.param<double>("dynamic_cluster_extractor", "peer_merge_distance_indoor", 1.0);
        spdlog::info("[cluster_extractor] peer merge distance (indoor): {}", peer_merge_distance);
    }
    

    track_match_distance = config.param<double>("dynamic_cluster_extractor", "track_match_distance", 1.5);
    track_match_iou      = config.param<double>("dynamic_cluster_extractor", "track_match_iou",      0.3);
    track_max_missed     = config.param<int>   ("dynamic_cluster_extractor", "track_max_missed",     3);
    min_dynamic_frames        = config.param<int>("dynamic_cluster_extractor", "min_dynamic_frames",        3);
    permanent_dynamic_frames  = config.param<int>("dynamic_cluster_extractor", "permanent_dynamic_frames",  10);
    permanent_static_frames   = config.param<int>("dynamic_cluster_extractor", "permanent_static_frames",   10);
    track_bbox_history_size   = config.param<int>("dynamic_cluster_extractor", "track_bbox_history_size",   5);
    use_motion_prediction         = config.param<bool>  ("dynamic_cluster_extractor", "use_motion_prediction",         true);
    fast_track_speed              = config.param<double>("dynamic_cluster_extractor", "fast_track_speed",              0.0);
    fast_track_min_dynamic_frames = config.param<int>   ("dynamic_cluster_extractor", "fast_track_min_dynamic_frames", 1);
    permanent_unlock_frames       = config.param<int>   ("dynamic_cluster_extractor", "permanent_unlock_frames",       2);
    use_centroid_association      = config.param<bool>  ("dynamic_cluster_extractor", "use_centroid_association",      true);
    assoc_gate_base               = config.param<double>("dynamic_cluster_extractor", "assoc_gate_base",               0.8);
    assoc_gate_per_range          = config.param<double>("dynamic_cluster_extractor", "assoc_gate_per_range",          0.03);
    assoc_gate_per_missed         = config.param<double>("dynamic_cluster_extractor", "assoc_gate_per_missed",         0.3);
    if (config.param<bool>("dynamic_cluster_extractor", "apply_lidar_imu_extrinsic", true)) {
        Config sensors(GlobalConfig::get_config_path("config_sensors"));
        T_imu_lidar = sensors.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity()).inverse();
    }
    release_static_frames         = config.param<int>   ("dynamic_cluster_extractor", "release_static_frames",         0);

    spdlog::debug("[cluster_extractor] eps_factor={:.2f} min_pts={} knn_max={} "
                  "min_cluster_voxels={} min_points_bbox={} "
                  "bbox_extent=[{:.2f},{:.2f}] bbox_volume=[{:.3f},{:.3f}] "
                  "cluster_iou_threshold={:.2f} peer_merge_distance={:.2f} "
                  "track_match_distance={:.2f} track_match_iou={:.2f} track_max_missed={} "
                  "min_dynamic_frames={} permanent_dynamic={} permanent_static={}",
                  eps_voxel_factor, min_pts, knn_max_neighbors,
                  min_cluster_voxels, min_points_for_bbox,
                  bbox_min_extent, bbox_max_extent,
                  bbox_min_volume, bbox_max_volume,
                  cluster_iou_threshold, peer_merge_distance,
                  track_match_distance, track_match_iou, track_max_missed,
                  min_dynamic_frames, permanent_dynamic_frames, permanent_static_frames);
}

DynamicClusterExtractorParams::~DynamicClusterExtractorParams() = default;

// ===========================================================================
// Constructors
// ===========================================================================

DynamicClusterExtractor::DynamicClusterExtractor(
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(), pose_kalman_filter_(pose_kalman_filter) {}

DynamicClusterExtractor::DynamicClusterExtractor(
    const DynamicClusterExtractorParams& params)
    : params_(params) {}

// ===========================================================================
// NMS utility
// ===========================================================================

static std::vector<BoundingBox> nms3D(
    const std::vector<BoundingBox>& boxes,
    double iou_threshold)
{
    const int N = static_cast<int>(boxes.size());
    if (N == 0) return {};

    struct Candidate { int idx; double score; };
    std::vector<Candidate> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& s = boxes[i].get_size();
        candidates.push_back({i, s.x() * s.y() * s.z()});
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b){ return a.score > b.score; });

    std::vector<bool> suppressed(N, false);
    std::vector<BoundingBox> result;
    result.reserve(N);
    for (int i = 0; i < N; ++i) {
        const int ii = candidates[i].idx;
        if (suppressed[ii]) continue;
        result.push_back(boxes[ii]);
        for (int j = i + 1; j < N; ++j) {
            const int jj = candidates[j].idx;
            if (suppressed[jj]) continue;
            // Track protection: two bboxes with distinct known tracks must stay separate.
            const int tid_ii = boxes[ii].get_track_id();
            const int tid_jj = boxes[jj].get_track_id();
            if (tid_jj != -1 && tid_ii != -1 && tid_jj != tid_ii) continue;
            if (boxes[ii].overlap(boxes[jj]) > iou_threshold)
                suppressed[jj] = true;
        }
    }
    return result;
}

// ===========================================================================
// extract_clusters()
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::extract_clusters(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap, double stamp)
{
    const auto cluster_map = cluster_voxels(voxelmap);

    int num_clusters = 0;
    for (int id : cluster_map)
        if (id >= num_clusters) num_clusters = id + 1;

    spdlog::debug("[cluster_extractor] found {} clusters", num_clusters);

    const auto clusters = build_point_clusters(voxelmap, cluster_map, num_clusters);
    auto bboxes = compute_bounding_boxes(clusters);

    if (pose_kalman_filter_) {
        const double dt = (last_stamp_ > 0.0 && stamp > last_stamp_) ? (stamp - last_stamp_) : 0.0;
        last_stamp_ = stamp;

        // getPose() = T_world_sensor. Maps previous-sensor-frame quantities into the
        // current sensor frame: T_cur^-1 * T_prev (was (T_cur * T_prev^-1)^-1, a world-frame delta).
        const Eigen::Isometry3d cur_pose     = pose_kalman_filter_->getPose() * params_.T_imu_lidar;   // T_world_lidar
        const Eigen::Isometry3d T_to_current = cur_pose.inverse() * last_pose_;
        last_pose_ = cur_pose;

        // Label bboxes with historical track state BEFORE NMS/merge so that
        // bboxes with distinct known tracks are protected from suppression/absorption.
        if (!tracks_.empty()) {
            label_bboxes_from_tracks(bboxes, T_to_current, dt);
        }

        bboxes = nms3D(bboxes, params_.cluster_iou_threshold);
        spdlog::debug("[cluster_extractor] {} bboxes after NMS", bboxes.size());

        bboxes = merge_nearby_clusters(bboxes);
        spdlog::debug("[cluster_extractor] {} bboxes after peer merge", bboxes.size());

        update_tracks(bboxes, T_to_current, dt);
    } else {
        bboxes = nms3D(bboxes, params_.cluster_iou_threshold);
        spdlog::debug("[cluster_extractor] {} bboxes after NMS", bboxes.size());
        bboxes = merge_nearby_clusters(bboxes);
        spdlog::debug("[cluster_extractor] {} bboxes after peer merge", bboxes.size());
    }

    return bboxes;
}

// ===========================================================================
// cluster_voxels()  —  DBSCAN on non-wall voxel centroids
// ===========================================================================

DynamicClusterExtractor::ClusterMap
DynamicClusterExtractor::cluster_voxels(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const
{
    const int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());

    ClusterMap cluster_map(nvox, -1);

    std::vector<int>             active_ids;
    std::vector<Eigen::Vector4d> active_cents;
    active_ids.reserve(nvox);
    active_cents.reserve(nvox);

    for (int i = 0; i < nvox; ++i) {
        const auto& v = voxelmap->lookup_voxel(i);
        if (!v.is_wall && !v.is_ground && !v.is_outlier) {
            active_ids.push_back(i);
            active_cents.push_back(v.mean);
        }
    }

    const int n_active = static_cast<int>(active_ids.size());
    spdlog::debug("[DBSCAN] nvox={} active(non-wall)={}", nvox, n_active);
    if (n_active == 0) return cluster_map;

    const double voxel_res = voxelmap->voxel_resolution();
    const double eps       = params_.eps_voxel_factor * voxel_res;
    const double eps2      = eps * eps;
    const int    min_pts   = params_.min_pts;
    const int    knn_k     = std::min(n_active, params_.knn_max_neighbors);

    spdlog::debug("[DBSCAN] voxel_res={:.3f} eps={:.3f} min_pts={} knn_k={}",
                  voxel_res, eps, min_pts, knn_k);

    gtsam_points::KdTree tree(active_cents.data(), n_active);

    auto range_query = [&](int local_idx) -> std::vector<int> {
        std::vector<size_t> k_idx(knn_k);
        std::vector<double> k_sq (knn_k);
        const size_t found = tree.knn_search(
            active_cents[local_idx].data(), knn_k, k_idx.data(), k_sq.data());

        std::vector<int> neighbors;
        neighbors.reserve(found);
        for (size_t i = 0; i < found; ++i)
            if (k_sq[i] <= eps2 && static_cast<int>(k_idx[i]) != local_idx)
                neighbors.push_back(static_cast<int>(k_idx[i]));
        return neighbors;
    };

    constexpr int UNVISITED = -2;
    constexpr int NOISE_LBL = -1;

    std::vector<int>  label(n_active, UNVISITED);
    std::vector<bool> in_seed(n_active, false);
    int cluster_id = 0;

    for (int i = 0; i < n_active; ++i) {
        if (label[i] != UNVISITED) continue;

        const auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts) {
            label[i] = NOISE_LBL;
            continue;
        }

        label[i] = cluster_id;
        std::vector<int> seed_set = neighbors;
        for (int nb : neighbors) in_seed[nb] = true;

        for (int s = 0; s < static_cast<int>(seed_set.size()); ++s) {
            const int q = seed_set[s];
            if (label[q] == NOISE_LBL) { label[q] = cluster_id; continue; }
            if (label[q] != UNVISITED)  continue;

            label[q] = cluster_id;
            const auto q_nbrs = range_query(q);
            if (static_cast<int>(q_nbrs.size()) >= min_pts) {
                for (int nb : q_nbrs) {
                    if ((label[nb] == UNVISITED || label[nb] == NOISE_LBL) && !in_seed[nb]) {
                        seed_set.push_back(nb);
                        in_seed[nb] = true;
                    }
                }
            }
        }

        for (int nb : seed_set) in_seed[nb] = false;
        in_seed[i] = false;
        ++cluster_id;
    }

    spdlog::debug("[DBSCAN] raw_clusters={}", cluster_id);

    std::vector<int> cluster_size(cluster_id, 0);
    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0) ++cluster_size[label[i]];

    std::vector<int> remap(cluster_id, -1);
    int valid_id = 0;
    for (int c = 0; c < cluster_id; ++c)
        if (cluster_size[c] >= params_.min_cluster_voxels)
            remap[c] = valid_id++;

    spdlog::debug("[DBSCAN] {} clusters after size filter (min_voxels={})",
                  valid_id, params_.min_cluster_voxels);

    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0 && remap[label[i]] >= 0)
            cluster_map[active_ids[i]] = remap[label[i]];

    return cluster_map;
}

// ===========================================================================
// build_point_clusters()
// ===========================================================================

std::vector<std::vector<Eigen::Vector4d>>
DynamicClusterExtractor::build_point_clusters(
    VoxelMapPtr       voxelmap,
    const ClusterMap& cluster_map,
    int               num_clusters) const
{
    std::vector<std::vector<Eigen::Vector4d>> point_clusters(num_clusters);

    const int nvox = static_cast<int>(cluster_map.size());
    for (int i = 0; i < nvox; ++i) {
        const int cid = cluster_map[i];
        if (cid < 0) continue;
        const auto& voxel = voxelmap->lookup_voxel(i);
        point_clusters[cid].insert(
            point_clusters[cid].end(),
            voxel.voxel_points.begin(),
            voxel.voxel_points.end());
    }

    return point_clusters;
}

// ===========================================================================
// compute_bounding_boxes()
// ===========================================================================

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters) const
{
    return compute_bounding_boxes(clusters, params_.min_points_for_bbox);
}

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters,
    int min_points) const
{
    std::vector<BoundingBox> boxes;
    boxes.reserve(clusters.size());

    int rejected_pts  = 0;
    int rejected_geom = 0;

    for (const auto& cluster : clusters) {
        if (static_cast<int>(cluster.size()) < min_points) {
            ++rejected_pts;
            continue;
        }

        BoundingBox bbox(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(),
                         Eigen::Matrix3d::Identity());
        if (!createAABB(cluster, bbox)) {
            ++rejected_geom;
            continue;
        }
        boxes.push_back(bbox);
    }

    spdlog::debug("[cluster_extractor] {} bboxes kept (rejected: {} pts, {} geom, min_pts={})",
                  boxes.size(), rejected_pts, rejected_geom, min_points);
    return boxes;
}

// ===========================================================================
// createAABB()  —  AABB (axis-aligned, single pass over points)
// ===========================================================================

bool DynamicClusterExtractor::createAABB(
    const std::vector<Eigen::Vector4d>& cluster,
    BoundingBox& out_bbox) const
{
    Eigen::Vector3d pt_min( std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max());
    Eigen::Vector3d pt_max(-std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max());

    for (const auto& p : cluster) {
        pt_min = pt_min.cwiseMin(p.head<3>());
        pt_max = pt_max.cwiseMax(p.head<3>());
    }

    const Eigen::Vector3d size   = pt_max - pt_min;
    const Eigen::Vector3d center = 0.5 * (pt_min + pt_max);

    const double min_dim = size.minCoeff();
    const double max_dim = size.maxCoeff();
    const double volume  = size.x() * size.y() * size.z();

    if (params_.bbox_min_extent > 0.0 && min_dim < params_.bbox_min_extent) return false;
    if (params_.bbox_max_extent < 1e8 && max_dim > params_.bbox_max_extent) return false;
    if (params_.bbox_min_volume > 0.0 && volume < params_.bbox_min_volume)  return false;
    if (params_.bbox_max_volume < 1e8 && volume > params_.bbox_max_volume)  return false;

    out_bbox = BoundingBox(size, center, Eigen::Matrix3d::Identity());
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : cluster) mean += p.head<3>();
    if (!cluster.empty()) out_bbox.set_centroid(mean / static_cast<double>(cluster.size()));
    return true;
}

// ===========================================================================
// compute_union_bbox()  —  union AABB of two bboxes, keeping a's local frame
// ===========================================================================

static BoundingBox compute_union_bbox(const BoundingBox& a, const BoundingBox& b)
{
    const Eigen::Matrix3d& R    = a.get_rotation();
    const Eigen::Vector3d& c_a  = a.get_center();
    const Eigen::Vector3d  he_a = a.get_size() * 0.5;

    const Eigen::Vector3d c_b_local = R.transpose() * (b.get_center() - c_a);
    const Eigen::Matrix3d R_b_local = R.transpose() * b.get_rotation();
    const Eigen::Vector3d he_b      = b.get_size() * 0.5;

    Eigen::Vector3d lo = -he_a;
    Eigen::Vector3d hi =  he_a;

    for (int sx : {-1, 1})
    for (int sy : {-1, 1})
    for (int sz : {-1, 1}) {
        const Eigen::Vector3d v = c_b_local +
            R_b_local * Eigen::Vector3d(sx * he_b.x(), sy * he_b.y(), sz * he_b.z());
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
    }

    BoundingBox u(hi - lo, c_a + R * (0.5 * (lo + hi)), R);
    const double va = std::max(a.get_size().prod(), 1e-9), vb = std::max(b.get_size().prod(), 1e-9);
    u.set_centroid((va * a.get_centroid() + vb * b.get_centroid()) / (va + vb));
    return u;
}

// ===========================================================================
// merge_nearby_clusters()
//
// Iteratively merge current-frame bboxes whose centers are within
// peer_merge_distance.  Converges when no more merges happen (handles chains
// A–B–C where A and C only become close after B is absorbed into A).
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::merge_nearby_clusters(
    const std::vector<BoundingBox>& bboxes) const
{
    if (params_.peer_merge_distance <= 0.0 || bboxes.empty()) return bboxes;

    const double d2_max = params_.peer_merge_distance * params_.peer_merge_distance;
    std::vector<BoundingBox> working(bboxes);

    bool any_merge = true;
    while (any_merge) {
        any_merge = false;
        const int W = static_cast<int>(working.size());

        // Process largest bboxes first so small ones are absorbed into larger ones.
        std::vector<int> order(W);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return working[a].get_size().prod() > working[b].get_size().prod();
        });

        std::vector<bool>        absorbed(W, false);
        std::vector<Eigen::Vector3d> centers(W);
        for (int i = 0; i < W; ++i) centers[i] = working[i].get_center();

        for (int ii = 0; ii < W; ++ii) {
            const int i = order[ii];
            if (absorbed[i]) continue;
            for (int jj = ii + 1; jj < W; ++jj) {
                const int j = order[jj];
                if (absorbed[j]) continue;
                if ((centers[i] - centers[j]).squaredNorm() <= d2_max) {
                    // Track protection: two bboxes with distinct known track IDs must stay separate.
                    const int  tid_i = working[i].get_track_id();
                    const int  tid_j = working[j].get_track_id();
                    if (tid_i != -1 && tid_j != -1 && tid_i != tid_j) continue;
                    const int  kept_tid = (tid_i != -1) ? tid_i : tid_j;
                    const bool kept_dyn = working[i].is_dynamic_bbox() || working[j].is_dynamic_bbox();
                    working[i] = compute_union_bbox(working[i], working[j]);
                    working[i].set_track_id(kept_tid);
                    working[i].set_dynamic(kept_dyn);
                    centers[i] = working[i].get_center();
                    absorbed[j] = true;
                    any_merge   = true;
                }
            }
        }

        std::vector<BoundingBox> tmp;
        tmp.reserve(W);
        for (int i = 0; i < W; ++i)
            if (!absorbed[i]) tmp.push_back(working[i]);
        working = std::move(tmp);
    }

    spdlog::debug("[merge_nearby_clusters] {} -> {} bboxes", bboxes.size(), working.size());
    return working;
}

// ===========================================================================
// label_bboxes_from_tracks()
//
// Read-only pre-pass: for each current-frame bbox, find the best-matching
// track from the previous frame and copy its is_dynamic label and track_id.
// Track state is never mutated.  The label is temporary — update_tracks()
// will re-assign it correctly based on hysteresis after NMS/merge.
// ===========================================================================

// Minimum-cost assignment (Hungarian / Kuhn-Munkres, O(n^3)) on a rectangular cost matrix.
// Returns for each row the assigned column or -1. Entries >= kInf are never accepted.
static constexpr double kInf = 1e9;
static std::vector<int> hungarian(const std::vector<std::vector<double>>& cost, int rows, int cols) {
    std::vector<int> out(rows, -1);
    if (rows == 0 || cols == 0) return out;
    const int n = std::max(rows, cols);
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, kInf));
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) a[i + 1][j + 1] = cost[i][j];
    std::vector<double> u(n + 1), v(n + 1); std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i; int j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity()); std::vector<char> used(n + 1, false);
        do {
            used[j0] = true; const int i0 = p[j0]; double delta = std::numeric_limits<double>::infinity(); int j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
            j0 = j1;
        } while (p[j0] != 0);
        do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    for (int j = 1; j <= n; ++j)
        if (p[j] >= 1 && p[j] <= rows && j <= cols && a[p[j]][j] < kInf) out[p[j] - 1] = j - 1;
    return out;
}

int DynamicClusterExtractor::required_dynamic_frames(const Track& t) const {
    if (params_.fast_track_speed > 0.0 &&
        std::hypot(t.velocity.x(), t.velocity.y()) > params_.fast_track_speed)
        return std::min(params_.fast_track_min_dynamic_frames, params_.min_dynamic_frames);
    return params_.min_dynamic_frames;
}

BoundingBox DynamicClusterExtractor::predicted_bbox(
    const Track& t, const Eigen::Isometry3d& T_to_current, double dt) const
{
    BoundingBox b = t.last_bbox;
    b.transform(T_to_current);
    if (params_.use_motion_prediction && dt > 0.0) {
        Eigen::Isometry3d shift = Eigen::Isometry3d::Identity();
        shift.translation() = (T_to_current.linear() * t.velocity) * dt;
        b.transform(shift);
    }
    return b;
}

void DynamicClusterExtractor::label_bboxes_from_tracks(
    std::vector<BoundingBox>& bboxes,
    const Eigen::Isometry3d&  T_to_current,
    double                    dt) const
{
    const double D2 = params_.track_match_distance * params_.track_match_distance;

    for (auto& bbox : bboxes) {
        double best_ov  = 0.0;
        bool   is_dyn   = false;
        int    best_tid = -1;

        for (const auto& t : tracks_) {
            const BoundingBox pred = predicted_bbox(t, T_to_current, dt);
            double score;
            if (params_.use_centroid_association) {
                const double gate = params_.assoc_gate_base + params_.assoc_gate_per_range * bbox.get_centroid().head<2>().norm()
                                  + params_.assoc_gate_per_missed * t.missed_frames;
                const double d = (pred.get_centroid() - bbox.get_centroid()).norm();
                if (d > gate) continue;
                score = 1.0 / (1e-3 + d);          // closer = better
            } else {
                if ((pred.get_center() - bbox.get_center()).squaredNorm() > D2) continue;
                score = pred.overlap(bbox);
                if (score < params_.track_match_iou) continue;
            }
            if (score <= best_ov) continue;

            best_ov  = score;
            best_tid = t.id;
            if (t.permanent_state == PermanentState::STATIC) {
                is_dyn = false;
            } else if (t.permanent_state == PermanentState::DYNAMIC) {
                is_dyn = true;
            } else {
                is_dyn = (t.dynamic_frames >= required_dynamic_frames(t));
            }
        }
        bbox.set_dynamic(is_dyn);
        bbox.set_track_id(best_tid);
    }
}

// ===========================================================================
// update_tracks()
//
// Overlap-based tracking with constant-velocity prediction.
// Matching criterion: predicted center distance < track_match_distance (cheap gate)
//                     AND overlap(predicted bbox, bbox) > track_match_iou.
// Greedy assignment ranked by overlap (best overlap first).
// ===========================================================================

void DynamicClusterExtractor::update_tracks(
    std::vector<BoundingBox>& bboxes,
    const Eigen::Isometry3d&  T_to_current,
    double                    dt)
{
    const int N_tracks = static_cast<int>(tracks_.size());
    const int N_bboxes = static_cast<int>(bboxes.size());

    // Predicted bboxes (computed from the previous-frame state).
    std::vector<BoundingBox> predicted;
    predicted.reserve(N_tracks);
    for (const auto& t : tracks_)
        predicted.push_back(predicted_bbox(t, T_to_current, dt));

    // Transform track state and history into current sensor frame.
    // (t.center holds the point centroid when use_centroid_association is enabled)
    for (auto& t : tracks_) {
        t.center   = T_to_current * t.center;
        t.velocity = T_to_current.linear() * t.velocity;
        t.last_bbox.transform(T_to_current);
        for (auto& hbbox : t.bbox_history)
            hbbox.transform(T_to_current);
    }


    // Build candidate pairs gated by distance, ranked by overlap.
    struct Pair { int t_idx, b_idx; double overlap; };
    std::vector<Pair> pairs;

    const double D2 = params_.track_match_distance * params_.track_match_distance;

    if (params_.use_centroid_association) {
        // Global assignment on predicted-centroid distance with a range/missed-dependent gate.
        std::vector<std::vector<double>> cost(N_tracks, std::vector<double>(N_bboxes, kInf));
        for (int t = 0; t < N_tracks; ++t)
            for (int b = 0; b < N_bboxes; ++b) {
                const double gate = params_.assoc_gate_base
                    + params_.assoc_gate_per_range * bboxes[b].get_centroid().head<2>().norm()
                    + params_.assoc_gate_per_missed * tracks_[t].missed_frames;
                const double d = (predicted[t].get_centroid() - bboxes[b].get_centroid()).norm();
                if (d <= gate) cost[t][b] = d;
            }
        // Solve independently on each connected component of the gating graph
        // (identical optimum, but O(sum k^3) instead of O(n^3) with ~hundreds of clusters outdoors).
        std::vector<int> parent(N_tracks + N_bboxes);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
        for (int t = 0; t < N_tracks; ++t)
            for (int b = 0; b < N_bboxes; ++b)
                if (cost[t][b] < kInf) parent[find(t)] = find(N_tracks + b);
        std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> comps;
        for (int t = 0; t < N_tracks; ++t) comps[find(t)].first.push_back(t);
        for (int b = 0; b < N_bboxes; ++b) comps[find(N_tracks + b)].second.push_back(b);
        for (const auto& [root, tb] : comps) {
            const auto& ts = tb.first; const auto& bs = tb.second;
            if (ts.empty() || bs.empty()) continue;
            std::vector<std::vector<double>> sub(ts.size(), std::vector<double>(bs.size()));
            for (size_t i = 0; i < ts.size(); ++i)
                for (size_t k = 0; k < bs.size(); ++k) sub[i][k] = cost[ts[i]][bs[k]];
            const auto assign = hungarian(sub, static_cast<int>(ts.size()), static_cast<int>(bs.size()));
            for (size_t i = 0; i < ts.size(); ++i)
                if (assign[i] >= 0) pairs.push_back({ts[i], bs[assign[i]], 1.0 / (1e-3 + sub[i][assign[i]])});
        }
    } else {
        for (int t = 0; t < N_tracks; ++t) {
            for (int b = 0; b < N_bboxes; ++b) {
                if ((predicted[t].get_center() - bboxes[b].get_center()).squaredNorm() > D2) continue;
                const double ov = predicted[t].overlap(bboxes[b]);
                if (ov >= params_.track_match_iou)
                    pairs.push_back({t, b, ov});
            }
        }
    }

    // Best score first for greedy assignment (no-op conflicts for Hungarian output).
    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b){ return a.overlap > b.overlap; });

    std::vector<bool> track_matched(N_tracks, false);
    std::vector<bool> bbox_matched (N_bboxes, false);

    for (const auto& p : pairs) {
        if (track_matched[p.t_idx] || bbox_matched[p.b_idx]) continue;
        track_matched[p.t_idx] = true;
        bbox_matched [p.b_idx] = true;

        Track& t = tracks_[p.t_idx];

        // Velocity estimation: displacement of the compensated center in m/s.
        // t.center / t.velocity are already expressed in the current frame.
        if (dt > 0.0) {
            const Eigen::Vector3d disp = (params_.use_centroid_association ? bboxes[p.b_idx].get_centroid()
                                                                           : bboxes[p.b_idx].get_center()) - t.center;
            const Eigen::Vector3d vel_meas = disp / dt;
            t.velocity = 0.6 * t.velocity + 0.4 * vel_meas;   // EMA
        }
        bboxes[p.b_idx].set_velocity(t.velocity);

        // Push previous bbox (already in current frame) to history before overwriting.
        if (params_.track_bbox_history_size > 0) {
            t.bbox_history.push_back(t.last_bbox);
            while (static_cast<int>(t.bbox_history.size()) > params_.track_bbox_history_size)
                t.bbox_history.pop_front();
        }

        t.center        = params_.use_centroid_association ? bboxes[p.b_idx].get_centroid() : bboxes[p.b_idx].get_center();
        // Permanent state overrides the hysteresis counter.
        if (t.permanent_state == PermanentState::DYNAMIC) {
            bboxes[p.b_idx].set_dynamic(true);
            bboxes[p.b_idx].set_locked(true);
        } else if (t.permanent_state == PermanentState::STATIC) {
            bboxes[p.b_idx].set_dynamic(false);
            bboxes[p.b_idx].set_locked(true);
        } else {
            bboxes[p.b_idx].set_dynamic(t.dynamic_frames >= required_dynamic_frames(t));
            bboxes[p.b_idx].set_locked(false);
        }
        t.last_bbox     = bboxes[p.b_idx];
        t.missed_frames = 0;
        t.age++;
        bboxes[p.b_idx].set_track_id(t.id);
       
        spdlog::debug("[tracker] matched track={} age={} overlap={:.3f} center=({:.2f},{:.2f},{:.2f})",
                      t.id, t.age, p.overlap,
                      t.center.x(), t.center.y(), t.center.z());
    }

    // Unmatched bboxes → new tracks.
    for (int b = 0; b < N_bboxes; ++b) {
        if (bbox_matched[b]) continue;
        Track t;
        t.id            = next_track_id_++;
        t.center        = params_.use_centroid_association ? bboxes[b].get_centroid() : bboxes[b].get_center();
        t.last_bbox     = bboxes[b];
        t.age           = 1;
        t.missed_frames = 0;
        tracks_.push_back(t);
        bboxes[b].set_track_id(t.id);
        bboxes[b].set_dynamic(false);  // New tracks start static (hysteresis).
        spdlog::debug("[tracker] new track={} at ({:.2f},{:.2f},{:.2f})",
                      t.id, t.center.x(), t.center.y(), t.center.z());
    }

    // Increment missed frames and prune dead tracks.
    // Unmatched tracks coast with constant velocity so they can be re-acquired later.
    for (int t = 0; t < N_tracks; ++t) {
        if (track_matched[t]) continue;
        Track& trk = tracks_[t];
        trk.missed_frames++;
        if (params_.use_motion_prediction && dt > 0.0) {
            Eigen::Isometry3d shift = Eigen::Isometry3d::Identity();
            shift.translation() = trk.velocity * dt;
            trk.center = shift * trk.center;
            trk.last_bbox.transform(shift);
        }
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const Track& trk){
                return trk.missed_frames > params_.track_max_missed;
            }),
        tracks_.end());

    spdlog::debug("[tracker] active_tracks={}", tracks_.size());
}

// ===========================================================================
// update_dynamic_feedback()
//
// Must be called after reject() each frame.  Matches post-rejection bboxes to
// tracks by track_id and updates the dynamic_frames hysteresis counter:
//   - bbox confirmed dynamic  → increment counter
//   - bbox not dynamic / not found → reset counter to 0
// ===========================================================================

void DynamicClusterExtractor::update_dynamic_feedback(
    const std::vector<BoundingBox>& post_rejection_bboxes)
{
    for (auto& track : tracks_) {
        // PERMANENT_DYNAMIC keeps its state for the track lifetime.
        // PERMANENT_STATIC is released after permanent_unlock_frames consecutive frames
        // of strong motion evidence (e.g. a parked car that starts moving).
        if (track.permanent_state == PermanentState::STATIC && params_.permanent_unlock_frames > 0) {
            bool strong = false;
            for (const auto& bbox : post_rejection_bboxes)
                if (bbox.get_track_id() == track.id) { strong = bbox.has_strong_motion(); break; }
            track.unlock_frames = strong ? track.unlock_frames + 1 : 0;
            if (track.unlock_frames >= params_.permanent_unlock_frames) {
                track.permanent_state = PermanentState::NONE;
                track.static_frames   = 0;
                track.dynamic_frames  = track.unlock_frames;   // strong motion counts as dynamic evidence
                track.unlock_frames   = 0;
                spdlog::debug("[tracker] track={} PERMANENT_STATIC released (strong motion)", track.id);
            }
            continue;
        }
        if (track.permanent_state != PermanentState::NONE) {
            spdlog::debug("[tracker] track={} PERMANENT_{}", track.id,
                          track.permanent_state == PermanentState::DYNAMIC ? "DYNAMIC" : "STATIC");
            continue;
        }

        // Find the matching post-rejection bbox by track_id.
        bool is_dynamic_this_frame = false;
        bool found = false;
        for (const auto& bbox : post_rejection_bboxes) {
            if (bbox.get_track_id() != track.id) continue;
            is_dynamic_this_frame = bbox.is_dynamic_bbox();
            found = true;
            break;
        }

        // Missed frame (occluded / merged cluster): reset both counters.
        // age does not increment on a missed frame, so counting it as static
        // evidence would let static_frames outrun age and trigger permanent
        // state on a track that has not yet been observed enough times.
        if (!found) {
            // Missed / occluded: keep the counters (the track coasts), unless the
            // legacy behaviour is requested.
            if (params_.release_static_frames <= 0) { track.static_frames = 0; track.dynamic_frames = 0; }
            continue;
        }

        if (is_dynamic_this_frame) {
            track.dynamic_frames++;
            track.static_frames = 0;
        } else {
            track.static_frames++;
            // Asymmetric hysteresis: a confirmed-dynamic track needs release_static_frames
            // consecutive static frames before it is switched back to static.
            const bool confirmed = track.dynamic_frames >= required_dynamic_frames(track);
            if (!confirmed || params_.release_static_frames <= 0 || track.static_frames >= params_.release_static_frames)
                track.dynamic_frames = 0;
        }

        // Check permanent state transitions (0 = feature disabled).
        if (params_.permanent_dynamic_frames > 0 &&
            track.dynamic_frames >= params_.permanent_dynamic_frames) {
            track.permanent_state = PermanentState::DYNAMIC;
            spdlog::debug("[tracker] track={} -> PERMANENTLY DYNAMIC (age={})",
                         track.id, track.age);
        } else if (params_.permanent_static_frames > 0 &&
                   track.static_frames >= params_.permanent_static_frames) {
            track.permanent_state = PermanentState::STATIC;
            spdlog::debug("[tracker] track={} -> PERMANENTLY STATIC (age={})",
                         track.id, track.age);
        }

        spdlog::debug("[tracker] track={} dyn={} sta={} perm={}",
                      track.id, track.dynamic_frames, track.static_frames,
                      track.permanent_state == PermanentState::NONE    ? "none"
                      : track.permanent_state == PermanentState::DYNAMIC ? "DYNAMIC"
                                                                          : "STATIC");
    }
}

// ===========================================================================
// get_dynamic_track_history()
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::get_dynamic_track_history() const {
    std::vector<BoundingBox> result;
    for (const auto& t : tracks_) {
        if (t.bbox_history.empty()) continue;
        const bool is_confirmed =
            (t.permanent_state == PermanentState::DYNAMIC) ||
            (t.permanent_state == PermanentState::NONE && t.dynamic_frames >= required_dynamic_frames(t));
        if (!is_confirmed) continue;
        for (const auto& hbbox : t.bbox_history)
            result.push_back(hbbox);
    }
    spdlog::debug("[cluster_extractor] get_dynamic_track_history: {} historical bboxes", result.size());
    return result;
}

} // namespace glim
