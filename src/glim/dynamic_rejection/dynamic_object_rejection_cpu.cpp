#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <spdlog/spdlog.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/config.hpp>

#include <glim/util/config.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif

#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

// ===========================================================================
// Helper: num_voxels() from a DynamicVoxelMapCPU
// ===========================================================================

static int nvox_of(const gtsam_points::DynamicVoxelMapCPU& vm) {
    return static_cast<int>(
        vm.gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());
}

// ===========================================================================
// DynamicObjectRejectionParamsCPU
// ===========================================================================

DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU() {
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));

    const char* k = "dynamic_object_rejection";
    dynamic_score_threshold        = config.param<double>(k, "dynamic_score_threshold",        0.08);
    tier1_threshold_factor         = config.param<double>(k, "tier1_threshold_factor",         0.55);
    unconstrained_threshold_factor = config.param<double>(k, "unconstrained_threshold_factor", 3.0);
    memory_threshold_factor        = config.param<double>(k, "memory_threshold_factor",        0.70);
    num_threads                    = config.param<int>   (k, "num_threads",                    4);
    w_shift                        = config.param<double>(k, "w_shift",                        1.2);
    w_mahalanobis                  = config.param<double>(k, "w_mahalanobis",                  0.0);
    w_cluster                      = config.param<double>(k, "w_cluster",                      0.0);
    w_history                      = config.param<double>(k, "w_history",                      0.3);
    w_history_dynamic              = config.param<double>(k, "w_history_dynamic",              0.0);  // unused
    w_neighbor                     = config.param<double>(k, "w_neighbor",                     0.2);
    history_factor                 = config.param<double>(k, "history_factor",                 0.3);  // unused
    frame_num_memory               = config.param<int>   (k, "frame_num_memory",               5);
    points_limit                   = config.param<double>(k, "points_limit",                   0.05);
    min_voxel_points               = config.param<int>   (k, "min_voxel_points",               -1);
    cluster_propagation_threshold  = config.param<double>(k, "cluster_propagation_threshold",  0.3);
    motion_threshold_scale         = config.param<double>(k, "motion_threshold_scale",         0.8);
    rotation_threshold_scale       = config.param<double>(k, "rotation_threshold_scale",       2.2);
    min_shift_m                    = config.param<double>(k, "min_shift_m",                    0.03);
    cluster_motion_scale           = config.param<double>(k, "cluster_motion_scale",           2.0);
    cluster_rotation_scale         = config.param<double>(k, "cluster_rotation_scale",         3.0);
    w_distance                     = config.param<double>(k, "w_distance",                     0.0);
    w_velocity                     = config.param<double>(k, "w_velocity",                     0.0);
    velocity_static_threshold      = config.param<double>(k, "velocity_static_threshold",      0.0);
    static_cluster_penalty_factor  = config.param<double>(k, "static_cluster_penalty_factor",  0.5);
    unmatched_dynamic_margin       = config.param<double>(k, "unmatched_dynamic_margin",       1.0);
    compare_baseline_frames        = config.param<int>   (k, "compare_baseline_frames",        5);
    noise_sigma0                   = config.param<double>(k, "noise_sigma0",                   0.045);
    noise_sigma_slope              = config.param<double>(k, "noise_sigma_slope",              0.007);
    noise_sigma_max                = config.param<double>(k, "noise_sigma_max",                0.10);
    noise_deadzone_k               = config.param<double>(k, "noise_deadzone_k",               3.0);
    keep_voxel_evidence            = config.param<bool>  (k, "keep_voxel_evidence",            false);
    unlock_ratio_factor            = config.param<double>(k, "unlock_ratio_factor",            2.0);
    evidence_enabled               = config.param<bool>  (k, "evidence_enabled",               false);
    evidence_resolution            = config.param<double>(k, "evidence_resolution",            0.4);
    evidence_decay                 = config.param<double>(k, "evidence_decay",                 0.7);
    evidence_gain                  = config.param<double>(k, "evidence_gain",                  1.0);
    evidence_on                    = config.param<double>(k, "evidence_on",                    1.5);
    evidence_off                   = config.param<double>(k, "evidence_off",                   0.5);

    // Frame history must hold at least the long-baseline frame.
    if (frame_num_memory < compare_baseline_frames)
        frame_num_memory = compare_baseline_frames;

    spdlog::debug("[dynamic_rejection] params loaded");
}

DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() = default;


// ===========================================================================
// Construction
// ===========================================================================

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(
    const DynamicObjectRejectionParamsCPU&  params,
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(params), pose_kalman_filter_(pose_kalman_filter), last_pose_(Eigen::Isometry3d::Identity())
{
    covariance_estimation_ = std::make_unique<CloudCovarianceEstimation>(params_.num_threads);
    inflate_params_ = VelocityInflationParams::from_config();

    if (!pose_kalman_filter_) {
        pose_kalman_filter_ = std::make_shared<PoseKalmanFilter>();
    }

    spdlog::info("[dynamic_rejection] inflate params: v_fwd_k={} v_rear_k={} v_lat_k={} v_min={}",
        inflate_params_.v_fwd_k, inflate_params_.v_rear_k,
        inflate_params_.v_lat_k, inflate_params_.v_min);
    spdlog::debug("[dynamic_rejection] DynamicObjectRejectionCPU constructed");
}


// ===========================================================================
// reject()  —  primary API
// ===========================================================================

DynamicRejectionResult DynamicObjectRejectionCPU::reject(
    const WallFilterResult&         wf_result,
    const PreprocessedFrame::Ptr&   source_frame,
    std::vector<BoundingBox>&       cluster_bboxes,
    const std::vector<BoundingBox>& historical_bboxes)
{
    spdlog::debug("[dynamic_rejection] reject begin: voxelmap={} wall_voxels={}/{}",
        static_cast<bool>(wf_result.voxelmap),
        wf_result.num_wall_voxels,
        wf_result.num_total_voxels);

    DynamicRejectionResult result;
    result.voxelmap = wf_result.voxelmap;

    // -----------------------------------------------------------------------
    // First frame: store as reference, return source unchanged
    // -----------------------------------------------------------------------
    if (voxelmap_history_.empty()) {
        spdlog::debug("[dynamic_rejection] first frame — storing reference voxelmap");
        voxelmap_history_.push_back(wf_result.voxelmap);
        pose_history_.push_back(Eigen::Isometry3d::Identity());  // no predecessor: identity delta
        last_pose_ = pose_kalman_filter_->getPose();
        last_dynamic_frame_ = nullptr;
        result.static_frame  = source_frame;
        result.dynamic_frame = nullptr;
        return result;
    }

    // getPose() returns T_world_sensor. Points of the current scan are expressed in the
    // current sensor frame, so the transform that maps them into the previous sensor
    // frame is T_prev^-1 * T_cur (NOT T_cur * T_prev^-1, which is a world-frame delta
    // and only coincides with the correct one for small rotations / poses near the origin).
    // NOTE: the pose is the IMU pose; the IMU-LiDAR extrinsic is assumed ~identity here.
    Eigen::Isometry3d cur_pose     = pose_kalman_filter_->getPose();
    Eigen::Isometry3d T_delta_pose = last_pose_.inverse() * cur_pose;
    last_pose_ = cur_pose;

    const double robot_translation = T_delta_pose.translation().norm();
    const double robot_rotation    = Eigen::AngleAxisd(T_delta_pose.rotation()).angle();
    translation_scale_ = params_.motion_threshold_scale    * robot_translation;
    rotation_scale_    = params_.rotation_threshold_scale  * robot_rotation;
    motion_scale_      = 1.0 + translation_scale_ + rotation_scale_;

    spdlog::debug("[dynamic_rejection] delta pose since last frame: translation=({:.4f},{:.4f},{:.4f}) motion_scale={:.3f}",
        T_delta_pose.translation().x(), T_delta_pose.translation().y(), T_delta_pose.translation().z(),
        motion_scale_);

    // -----------------------------------------------------------------------
    // Score voxels against previous frame, then propagate to neighbours
    // -----------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    index_voxel_bboxes(*wf_result.voxelmap, cluster_bboxes);
    score_voxels(*wf_result.voxelmap, *voxelmap_history_.back(), cluster_bboxes, T_delta_pose);
    auto t1 = std::chrono::steady_clock::now();
    spdlog::debug("[dynamic_rejection] score_voxels took {:.1f} ms",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    const int nvox = nvox_of(*wf_result.voxelmap);
    propagate_to_neighbors(*wf_result.voxelmap, nvox);
    propagate_to_clusters(*wf_result.voxelmap, cluster_bboxes, historical_bboxes);
    if (params_.evidence_enabled) apply_world_evidence(*wf_result.voxelmap, cur_pose);

    // -----------------------------------------------------------------------
    // Split voxel points into static / dynamic buckets
    // -----------------------------------------------------------------------
    std::vector<Eigen::Vector4d> static_pts,  dynamic_pts;
    std::vector<double>          static_int,  dynamic_int;
    std::vector<double>          static_tim,  dynamic_tim;

    collect_points(*wf_result.voxelmap,
                   static_pts,  static_int,  static_tim,
                   dynamic_pts, dynamic_int, dynamic_tim);

    // -----------------------------------------------------------------------
    // Update history ring (voxelmap_history_ and pose_history_ stay aligned)
    // -----------------------------------------------------------------------
    voxelmap_history_.push_back(wf_result.voxelmap);
    pose_history_.push_back(T_delta_pose);
    while (static_cast<int>(voxelmap_history_.size()) > params_.frame_num_memory) {
        voxelmap_history_.pop_front();
        pose_history_.pop_front();
    }

    // -----------------------------------------------------------------------
    // Build output frames
    // -----------------------------------------------------------------------
    if (static_pts.empty()) {
        spdlog::warn("[dynamic_rejection] no static points — returning source frame unchanged");
        result.static_frame = source_frame;
    } else {
        result.static_frame = build_frame(
            std::move(static_pts), std::move(static_int), std::move(static_tim),
            *source_frame);
    }

    if (!dynamic_pts.empty()) {
        auto dyn           = std::make_shared<PreprocessedFrame>();
        dyn->stamp         = source_frame->stamp;
        dyn->scan_end_time = source_frame->scan_end_time;
        dyn->points        = std::move(dynamic_pts);
        dyn->intensities   = std::move(dynamic_int);
        dyn->times         = std::move(dynamic_tim);
        dyn->k_neighbors   = 0;
        last_dynamic_frame_  = dyn;
        result.dynamic_frame = dyn;
    } else {
        last_dynamic_frame_  = nullptr;
        result.dynamic_frame = nullptr;
    }

    spdlog::debug("[dynamic_rejection] reject done: static={} dynamic={}",
        result.static_frame  ? result.static_frame->points.size()  : 0,
        result.dynamic_frame ? result.dynamic_frame->points.size() : 0);

    return result;
}


// ===========================================================================
// Parallel-for helper (OpenMP / TBB / serial, same back-end selection as before)
// ===========================================================================

template <typename F>
static void parallel_for_voxels(int n, int num_threads, const F& f) {
    if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
        for (int j = 0; j < n; ++j) f(j);
    } else {
#ifdef GTSAM_POINTS_USE_TBB
        tbb::parallel_for(
            tbb::blocked_range<int>(0, n, 16),
            [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j < r.end(); ++j) f(j);
            });
#else
        for (int j = 0; j < n; ++j) f(j);
#endif
    }
}


// ===========================================================================
// index_voxel_bboxes()
// ===========================================================================
// Single pass voxel x bbox, reused by score_voxels() and propagate_to_clusters()
// (previously the same containment tests were repeated three times, twice serially).

void DynamicObjectRejectionCPU::index_voxel_bboxes(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    const std::vector<BoundingBox>&         cluster_bboxes)
{
    const int nvox       = nvox_of(voxelmap);
    const int n_clusters = static_cast<int>(cluster_bboxes.size());

    voxel_bboxes_.assign(nvox, {});
    if (n_clusters == 0) return;

    parallel_for_voxels(nvox, params_.num_threads, [&](int j) {
        const auto& mean = voxelmap.lookup_voxel(j).mean;
        auto& out = voxel_bboxes_[j];
        for (int c = 0; c < n_clusters; ++c) {
            const auto& b = cluster_bboxes[c];
            if (b.contains(mean)) {
                out.emplace_back(c, true);
            } else if (b.is_dynamic_bbox() && b.contains_inflated(mean, inflate_params_)) {
                out.emplace_back(c, false);
            }
        }
    });
}


// ===========================================================================
// match_voxel()
// ===========================================================================

int DynamicObjectRejectionCPU::match_voxel(
    const gtsam_points::DynamicVoxelMapCPU& map,
    const Eigen::Vector3d&                  p,
    double                                  voxel_res) const
{
    const Eigen::Vector4d p4(p.x(), p.y(), p.z(), 1.0);
    const auto coord = map.voxel_coord(p4);
    int idx = map.lookup_voxel_index(coord);
    if (idx >= 0) return idx;

    // Fallback: if exact hash misses (pose error >= 0.5 * voxel_size),
    // search the 26 neighbouring cells and pick the nearest centroid.
    const double max_dist2 = (2.0 * voxel_res) * (2.0 * voxel_res);
    double best_dist2 = max_dist2;
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const Eigen::Vector3i nc(coord.x() + dx, coord.y() + dy, coord.z() + dz);
        const int ni = map.lookup_voxel_index(nc);
        if (ni < 0) continue;
        const double d2 = (p - map.lookup_voxel(ni).mean.head<3>()).squaredNorm();
        if (d2 < best_dist2) { best_dist2 = d2; idx = ni; }
    }
    return idx;
}


// ===========================================================================
// score_voxels()
// ===========================================================================

void DynamicObjectRejectionCPU::score_voxels(
    gtsam_points::DynamicVoxelMapCPU&       current,
    const gtsam_points::DynamicVoxelMapCPU& previous,
    const std::vector<BoundingBox>&         cluster_bboxes,
    const Eigen::Isometry3d&                T_delta_pose)
{
    const int    nvox       = nvox_of(current);
    const double voxel_res  = current.voxel_resolution();
    const double inv_res    = 1.0 / voxel_res;
    // Minimum raw points per voxel. Legacy formula kept for config compatibility
    // (points_limit * voxel_res * 100, e.g. 0.05 * 0.25 * 100 = 1.25 points).
    const double points_thr = params_.min_voxel_points >= 0
        ? static_cast<double>(params_.min_voxel_points)
        : params_.points_limit * voxel_res * 100.0;

    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();

    const int hist_size  = static_cast<int>(voxelmap_history_.size());
    const int hist_depth = std::min(params_.frame_num_memory, hist_size) - 1;

    // Baseline frame t-K lives at voxelmap_history_[hist_size - K]
    // (voxelmap_history_.back() is t-1). Centroid noise between two scans is ~constant
    // w.r.t. the time gap (~6-9 cm on indoor data), while a moving object's
    // displacement grows linearly, so comparing against t-K raises the SNR ~K times.
    const int K = std::max(1, std::min(params_.compare_baseline_frames, hist_size));
    const auto& base_map = *voxelmap_history_[hist_size - K];

    std::mutex dyn_mutex;

    const auto mark_dynamic = [&](int j, const Eigen::Vector4d& mean) {
        const auto nbrs = get_neighbor_voxels(current, mean);
        std::lock_guard<std::mutex> lock(dyn_mutex);
        dynamic_voxels.insert(dynamic_voxels.end(), nbrs.size(), j);
        dynamic_voxels_neighbor_indices_.insert(
            dynamic_voxels_neighbor_indices_.end(), nbrs.begin(), nbrs.end());
    };

    const auto per_voxel = [&](int j) {
        auto& cur = current.lookup_voxel(j);
        cur.is_dynamic    = false;
        cur.dynamic_score = 0.0;

        if (cur.is_wall || cur.is_ground || cur.is_outlier) return;
        if (cur.num_points < points_thr) return;

        // ---- Cluster bbox check (base AABB + velocity-inflated ellipsoid) ----
        // Dynamic-wins policy: if a voxel overlaps both a static and a dynamic
        // bbox, the dynamic match takes priority.
        bool   in_any_bbox         = false;
        bool   in_any_dynamic_bbox = false;
        double best_cluster_speed  = 0.0;
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            (void)in_base;
            in_any_bbox = true;
            if (cluster_bboxes[c].is_dynamic_bbox()) in_any_dynamic_bbox = true;
            best_cluster_speed = std::max(best_cluster_speed, cluster_bboxes[c].get_velocity().norm());
        }
        const bool possibly_dynamic = in_any_dynamic_bbox;
        if (in_any_dynamic_bbox) {
            cur.dynamic_score += params_.w_cluster;
        } else {
            cur.dynamic_score -= params_.w_cluster * params_.static_cluster_penalty_factor;
        }
        if (params_.w_velocity > 0.0 && in_any_bbox)
            cur.dynamic_score += params_.w_velocity * (best_cluster_speed - params_.velocity_static_threshold);

        // ---- Transform centroid into previous frame and match ----
        const Eigen::Vector3d p_trans = T_delta_pose * cur.mean.head<3>();
        const Eigen::Vector4d p_trans4(p_trans.x(), p_trans.y(), p_trans.z(), 1.0);
        const int prev_idx = match_voxel(previous, p_trans, voxel_res);

        // ---- 0. Recent-dynamic memory (t-1, threshold reduction only, no score bonus) ----
        const bool was_recently_dynamic = prev_idx >= 0 && previous.lookup_voxel(prev_idx).is_dynamic;

        // ---- Move the centroid into the baseline frame t-K and match ----
        Eigen::Vector3d p_base = p_trans;                               // frame t-1
        for (int idx = hist_size - 1; idx > hist_size - K; --idx)
            p_base = pose_history_[idx] * p_base;                       // frame idx -> idx-1
        const int base_idx = (K == 1) ? prev_idx : match_voxel(base_map, p_base, voxel_res);

        if (base_idx < 0) {
            // Tier-1 (confirmed dynamic bbox): the object likely moved out of the
            // search window -> dynamic. Otherwise inconclusive -> static.
            if (possibly_dynamic) {
                cur.is_dynamic    = true;
                cur.dynamic_score = params_.dynamic_score_threshold + params_.unmatched_dynamic_margin;
                mark_dynamic(j, cur.mean);
            }
            return;
        }

        const auto& base = base_map.lookup_voxel(base_idx);

        // ---- 1. Centroid shift with range-dependent dead zone ----
        // Dead zone = noise_deadzone_k * sigma(r), sigma(r) = expected centroid noise
        // at range r (measured: ~5 cm at 1.5 m, ~9 cm beyond 6 m). Replaces the
        // linear w_distance penalty, which made walking people undetectable beyond ~3 m.
        const Eigen::Vector3d delta = p_base - base.mean.head<3>();
        const double range = cur.mean.head<3>().norm();
        const double sigma = std::min(params_.noise_sigma_max, params_.noise_sigma0 + params_.noise_sigma_slope * range);
        const double dead  = std::max(params_.min_shift_m, params_.noise_deadzone_k * sigma);
        const double shift = std::max(0.0, delta.norm() - dead) * inv_res;

        // ---- 2. Mahalanobis distance ----
        double mahal = 0.0;
        if (params_.w_mahalanobis > 0.0) {
            Eigen::Matrix3d cov = base.cov.topLeftCorner<3, 3>();
            cov = 0.5 * (cov + cov.transpose());
            cov.diagonal().array() += 1e-6;

            Eigen::LDLT<Eigen::Matrix3d> ldlt(cov);
            if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                const Eigen::Vector3d sol = ldlt.solve(delta);
                if (sol.allFinite()) {
                    const double q = delta.dot(sol);
                    if (q >= 0.0) mahal = std::sqrt(q);
                }
            }
        }

        cur.dynamic_score += params_.w_shift * shift + params_.w_mahalanobis * mahal;

        // ---- History suppression ----
        // Count only frames where the voxel was present AND static.
        if (hist_depth > 0) {
            int static_count = 0;
            Eigen::Vector4d cur_mean = p_trans4;   // expressed in frame of voxelmap_history_.back()

            for (int k = 0; k < hist_depth; ++k) {
                const int idx_hist = hist_size - 1 - k;
                const auto& past   = voxelmap_history_[idx_hist];

                const int voxel_idx = past->lookup_voxel_index(past->voxel_coord(cur_mean));
                if (voxel_idx >= 0 && !past->lookup_voxel(voxel_idx).is_dynamic)
                    ++static_count;

                // Move the point into the frame of voxelmap_history_[idx_hist - 1].
                const Eigen::Vector3d tmp = pose_history_[idx_hist] * cur_mean.head<3>();
                cur_mean = Eigen::Vector4d(tmp.x(), tmp.y(), tmp.z(), 1.0);
            }

            const double static_ratio = static_cast<double>(static_count) / hist_depth;
            cur.dynamic_score -= params_.w_history * static_ratio;
        }

        // ---- Distance penalty (suppresses far-range false positives) ----
        if (params_.w_distance > 0.0) {
            cur.dynamic_score -= params_.w_distance * cur.mean.head<3>().norm();
        }

        // ---- Threshold (three-tier, cluster-gated with dynamic memory) ----
        //  Tier 1 — inside a CONFIRMED dynamic cluster bbox:  base * tier1_threshold_factor
        //  Tier 2 — dynamic in the previous frame, no bbox:   base * memory_threshold_factor
        //  Tier 3 — isolated voxel:                           base * unconstrained_threshold_factor
        const double base_threshold = params_.dynamic_score_threshold * motion_scale_;
        double eff_threshold;
        if (possibly_dynamic) {
            eff_threshold = base_threshold * params_.tier1_threshold_factor;
        } else if (was_recently_dynamic) {
            eff_threshold = base_threshold * params_.memory_threshold_factor;
        } else {
            eff_threshold = base_threshold * params_.unconstrained_threshold_factor;
        }

        if (cur.dynamic_score > eff_threshold) {
            cur.is_dynamic = true;
            mark_dynamic(j, cur.mean);
        }
    };

    parallel_for_voxels(nvox, params_.num_threads, per_voxel);
}


// ===========================================================================
// apply_world_evidence()
// ===========================================================================
// Per-cell evidence in the world frame: e <- e * decay^(frames since last update)
// (+ gain if a voxel in the cell is dynamic this frame). A cell turns ON at
// evidence_on and OFF below evidence_off. Voxels take the state of their cell, so
// the label no longer depends on the cluster/track ID staying stable.

void DynamicObjectRejectionCPU::apply_world_evidence(
    gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor)
{
    ++frame_idx_;
    const int    nvox = nvox_of(voxelmap);
    const double inv  = 1.0 / params_.evidence_resolution;
    const auto key_of = [&](const Eigen::Vector3d& pw) {
        const int64_t x = static_cast<int64_t>(std::floor(pw.x() * inv)) & 0x1FFFFF;
        const int64_t y = static_cast<int64_t>(std::floor(pw.y() * inv)) & 0x1FFFFF;
        const int64_t z = static_cast<int64_t>(std::floor(pw.z() * inv)) & 0x1FFFFF;
        return (x << 42) | (y << 21) | z;
    };

    // 1) accumulate: one update per cell per frame
    std::vector<int64_t> keys(nvox, -1);
    std::unordered_map<int64_t, bool> hit;   // cell -> any dynamic voxel this frame
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (v.is_wall || v.is_ground || v.is_outlier) continue;
        keys[j] = key_of(T_world_sensor * v.mean.head<3>());
        auto& h = hit[keys[j]];
        h = h || v.is_dynamic;
    }
    for (const auto& [k, dyn] : hit) {
        auto& c = evidence_[k];
        c.e = static_cast<float>(c.e * std::pow(params_.evidence_decay, frame_idx_ - c.last));
        c.last = frame_idx_;
        if (dyn) c.e += static_cast<float>(params_.evidence_gain);
        if (!c.on && c.e >= params_.evidence_on) c.on = true;
        else if (c.on && c.e < params_.evidence_off) c.on = false;
    }
    // 2) relabel
    for (int j = 0; j < nvox; ++j)
        if (keys[j] >= 0) voxelmap.lookup_voxel(j).is_dynamic = evidence_[keys[j]].on;

    // 3) prune stale cells
    if (frame_idx_ % 50 == 0) {
        for (auto it = evidence_.begin(); it != evidence_.end();) {
            const double e = it->second.e * std::pow(params_.evidence_decay, frame_idx_ - it->second.last);
            it = (e < 0.05) ? evidence_.erase(it) : std::next(it);
        }
    }
}


// ===========================================================================
// propagate_to_neighbors()
// ===========================================================================

void DynamicObjectRejectionCPU::propagate_to_neighbors(
    gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox)
{
    for (int idx : dynamic_voxels_neighbor_indices_) {
        if (idx < 0 || idx >= nvox) continue;

        auto& v = voxelmap.lookup_voxel(idx);
        if (v.is_dynamic || v.is_wall || v.is_ground || v.is_outlier) continue;

        v.dynamic_score += params_.w_neighbor;
        if (v.dynamic_score > params_.dynamic_score_threshold * motion_scale_) {
            v.is_dynamic = true;
        }
    }
    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();
}


// ===========================================================================
// propagate_to_clusters()
// ===========================================================================
// If a percentage of voxels in a cluster are dynamic, mark the whole cluster.

void DynamicObjectRejectionCPU::propagate_to_clusters(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    std::vector<BoundingBox>&         cluster_bboxes,
    const std::vector<BoundingBox>&   historical_bboxes)
{
    if (cluster_bboxes.empty() && historical_bboxes.empty() && !params_.keep_voxel_evidence) return;

    const int nvox       = nvox_of(voxelmap);
    const int n_clusters = static_cast<int>(cluster_bboxes.size());

    std::vector<int> dynamic_count(n_clusters, 0);
    std::vector<int> total_count  (n_clusters, 0);

    // Dynamic-wins multi-cluster assignment: a voxel contributes to ALL base bboxes containing it.
    for (int j = 0; j < nvox; ++j) {
        const bool dyn = voxelmap.lookup_voxel(j).is_dynamic;
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            if (!in_base) continue;
            ++total_count[c];
            if (dyn) ++dynamic_count[c];
        }
    }

    // Scale the propagation threshold with robot motion.
    const double eff_prop_threshold =
        params_.cluster_propagation_threshold
        * std::pow(1.0 + translation_scale_, params_.cluster_motion_scale)
        * std::pow(1.0 + rotation_scale_,    params_.cluster_rotation_scale);
    const double unlock_threshold = eff_prop_threshold * params_.unlock_ratio_factor;

    // Hysteresis-gated state BEFORE this frame's ratio (used for voxel assignment).
    std::vector<bool> was_dynamic(n_clusters);
    for (int c = 0; c < n_clusters; ++c)
        was_dynamic[c] = cluster_bboxes[c].is_dynamic_bbox();

    for (int c = 0; c < n_clusters; ++c) {
        cluster_bboxes[c].set_strong_motion(false);
        if (total_count[c] == 0) continue;

        const double ratio = static_cast<double>(dynamic_count[c]) / total_count[c];
        // Strong-motion evidence is computed for locked clusters too, so the tracker
        // can release a PERMANENT_STATIC track that starts moving.
        cluster_bboxes[c].set_strong_motion(ratio > unlock_threshold);

        if (cluster_bboxes[c].is_locked()) {
            spdlog::debug("[dynamic_rejection] cluster {} LOCKED -> {} (ratio {:.1f}%, strong={})",
                          c, cluster_bboxes[c].is_dynamic_bbox() ? "DYNAMIC" : "static",
                          ratio * 100.0, cluster_bboxes[c].has_strong_motion());
            continue;
        }

        const bool is_dyn = ratio > eff_prop_threshold;
        cluster_bboxes[c].set_dynamic(is_dyn);  // feedback for update_dynamic_feedback()
        spdlog::debug("[dynamic_rejection] cluster {}: {}/{} dynamic ({:.1f}%) threshold={:.2f} -> {} (was {})",
                      c, dynamic_count[c], total_count[c], ratio * 100.0, eff_prop_threshold,
                      is_dyn ? "DYNAMIC" : "static", was_dynamic[c] ? "DYNAMIC" : "static");
    }

    // Final voxel assignment.
    parallel_for_voxels(nvox, params_.num_threads, [&](int j) {
        auto& v = voxelmap.lookup_voxel(j);
        // Structural voxels are never removed (ground under a person must stay in the map).
        if (v.is_wall || v.is_ground || v.is_outlier) { v.is_dynamic = false; return; }

        bool in_any_base = false, in_dynamic = false;
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            if (!was_dynamic[c]) {
                if (in_base) in_any_base = true;
                continue;
            }
            // Confirmed-dynamic bbox: base volume or velocity-inflated zone.
            in_dynamic = true;
            break;
        }

        if (in_dynamic) { v.is_dynamic = true; return; }
        if (in_any_base) { v.is_dynamic = false; return; }   // inside a static cluster only

        // Outside all bboxes: historical bboxes of confirmed-dynamic tracks.
        for (const auto& hbbox : historical_bboxes) {
            if (hbbox.contains(v.mean) || hbbox.contains_inflated(v.mean, inflate_params_)) {
                v.is_dynamic = true;
                return;
            }
        }

        // Per-voxel evidence (Tier 2/3 + neighbour propagation). Previously this was
        // always discarded; keep it when enabled.
        v.is_dynamic = params_.keep_voxel_evidence && v.is_dynamic;
    });
}


// ===========================================================================
// collect_points()
// ===========================================================================

void DynamicObjectRejectionCPU::collect_points(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    std::vector<Eigen::Vector4d>& static_pts,
    std::vector<double>&          static_int,
    std::vector<double>&          static_tim,
    std::vector<Eigen::Vector4d>& dynamic_pts,
    std::vector<double>&          dynamic_int,
    std::vector<double>&          dynamic_tim) const
{
    // Pre-size buckets to avoid repeated reallocations                  // OPT
    const int nvox = nvox_of(voxelmap);
    {
        size_t total_pts = 0;
        for (int j = 0; j < nvox; ++j)
            total_pts += voxelmap.lookup_voxel(j).voxel_points.size();
        static_pts.reserve(total_pts);
        static_int.reserve(total_pts);
        static_tim.reserve(total_pts);
        // Dynamic is typically small — reserve a fraction
        dynamic_pts.reserve(total_pts / 8);
        dynamic_int.reserve(total_pts / 8);
        dynamic_tim.reserve(total_pts / 8);
    }

    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);

        auto& pts  = v.is_dynamic ? dynamic_pts : static_pts;
        auto& ints = v.is_dynamic ? dynamic_int : static_int;
        auto& tims = v.is_dynamic ? dynamic_tim : static_tim;

        pts.insert(pts.end(),   v.voxel_points.begin(),      v.voxel_points.end());
        ints.insert(ints.end(), v.voxel_intensities.begin(), v.voxel_intensities.end());
        tims.insert(tims.end(), v.voxel_times.begin(),       v.voxel_times.end());
    }
}


// ===========================================================================
// build_frame()
// ===========================================================================

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::build_frame(
    std::vector<Eigen::Vector4d> points,
    std::vector<double>          intensities,
    std::vector<double>          times,
    const PreprocessedFrame&     source) const
{
    auto out           = std::make_shared<PreprocessedFrame>();
    out->stamp         = source.stamp;
    out->scan_end_time = source.scan_end_time;
    out->k_neighbors   = source.k_neighbors;
    out->points        = std::move(points);
    out->intensities   = std::move(intensities);
    out->times         = std::move(times);

    if (source.k_neighbors > 0 && !out->points.empty()) {
        out->neighbors = find_neighbors(
            out->points.data(),
            static_cast<int>(out->points.size()),
            source.k_neighbors);
    }

    return out;
}


// ===========================================================================
// find_neighbors()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::find_neighbors(
    const Eigen::Vector4d* points, int num_points, int k) const
{
    if (!points || num_points <= 0 || k <= 0) return {};

    gtsam_points::KdTree tree(points, num_points);
    std::vector<int>     neighbors(num_points * k);

    const auto per_point = [&](int i) {
        std::vector<size_t> k_idx(k, static_cast<size_t>(i));
        std::vector<double> k_sq(k);
        const size_t found =
            tree.knn_search(points[i].data(), k, k_idx.data(), k_sq.data());
        std::copy(k_idx.begin(), k_idx.begin() + found,
                  neighbors.begin() + i * k);
    };

    if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params_.num_threads) schedule(guided, 8)
        for (int i = 0; i < num_points; ++i) per_point(i);
    } else {
#ifdef GTSAM_POINTS_USE_TBB
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_points, 8),
            [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i < r.end(); ++i) per_point(i);
            });
#else
        for (int i = 0; i < num_points; ++i) per_point(i);
#endif
    }

    return neighbors;
}


// ===========================================================================
// get_neighbor_voxels()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::get_neighbor_voxels(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    const Eigen::Vector4d& mean) const
{
    std::vector<int> neighbors;
    neighbors.reserve(26);                                               // OPT: 3³-1 = 26 max neighbours
    const auto base = voxelmap.voxel_coord(mean);

    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        Eigen::Vector3i c(base.x() + dx, base.y() + dy, base.z() + dz); // OPT: avoid copy + 3 separate increments
        const int idx = voxelmap.lookup_voxel_index(c);
        if (idx >= 0) neighbors.push_back(idx);
    }

    return neighbors;
}

}  // namespace glim