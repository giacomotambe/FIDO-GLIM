#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <limits>

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
    sparse_voxel_mode              = config.param<int>   (k, "sparse_voxel_mode",              1);
    min_voxel_points               = config.param<int>   (k, "min_voxel_points",               -1);
    cluster_propagation_threshold  = config.param<double>(k, "cluster_propagation_threshold",  0.45);
    motion_threshold_scale         = config.param<double>(k, "motion_threshold_scale",         0.0);
    rotation_threshold_scale       = config.param<double>(k, "rotation_threshold_scale",       0.0);
    min_shift_m                    = config.param<double>(k, "min_shift_m",                    0.03);
    cluster_motion_scale           = config.param<double>(k, "cluster_motion_scale",           0.0);
    cluster_rotation_scale         = config.param<double>(k, "cluster_rotation_scale",         0.0);
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
    hybrid_enabled                 = config.param<bool>  (k, "hybrid_enabled",                 true);
    hybrid_min_range               = config.param<double>(k, "hybrid_min_range",               20.0);
    hybrid_max_range               = config.param<double>(k, "hybrid_max_range",               40.0);
    hybrid_max_points              = config.param<int>   (k, "hybrid_max_points",              8000);
    ground_band_m                  = config.param<double>(k, "ground_band_m",                  0.25);
    ground_cell_m                  = config.param<double>(k, "ground_cell_m",                  2.0);
    min_obj_height                 = config.param<double>(k, "min_obj_height",                 0.5);
    min_obj_speed                  = config.param<double>(k, "min_obj_speed",                  0.3);
    frame_max_dynamic_frac         = config.param<double>(k, "frame_max_dynamic_frac",         0.10);
    inflated_requires_evidence     = config.param<bool>  (k, "inflated_requires_evidence",     true);
    fast_confirm_ratio_factor      = config.param<double>(k, "fast_confirm_ratio_factor",      2.0);
    fast_confirm_min_vis           = config.param<double>(k, "fast_confirm_min_vis",           0.2);
    moving_vis_alt                 = config.param<double>(k, "moving_vis_alt",                 0.3);
    point_refine_enabled           = config.param<bool>  (k, "point_refine_enabled",           false);
    point_grow_radius0             = config.param<double>(k, "point_grow_radius0",             0.2);
    point_grow_radius_k            = config.param<double>(k, "point_grow_radius_k",            0.02);
    point_grow_radius_max          = config.param<double>(k, "point_grow_radius_max",          0.5);
    point_bbox_margin              = config.param<double>(k, "point_bbox_margin",              0.3);
    point_static_veto              = config.param<bool>  (k, "point_static_veto",              false);
    point_knn                      = config.param<int>   (k, "point_knn",                      16);
    point_grow_on_visibility       = config.param<bool>  (k, "point_grow_on_visibility",       false);
    foot_min_hag                   = config.param<double>(k, "foot_min_hag",                   -1.0);
    apply_lidar_imu_extrinsic      = config.param<bool>  (k, "apply_lidar_imu_extrinsic",      true);
    if (apply_lidar_imu_extrinsic) {
        Config sensors(GlobalConfig::get_config_path("config_sensors"));
        T_imu_lidar = sensors.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity()).inverse();
    }
    pose_err_trans_k               = config.param<double>(k, "pose_err_trans_k",               0.01);
    pose_err_rot_k                 = config.param<double>(k, "pose_err_rot_k",                 0.01);
    visibility_enabled             = config.param<bool>  (k, "visibility_enabled",             true);
    w_visibility                   = config.param<double>(k, "w_visibility",                   0.5);
    visibility_res_deg             = config.param<double>(k, "visibility_res_deg",             0.7);
    visibility_abs_thr             = config.param<double>(k, "visibility_abs_thr",             0.3);
    visibility_rel_thr             = config.param<double>(k, "visibility_rel_thr",             0.03);
    visibility_min_age             = config.param<int>   (k, "visibility_min_age",             3);
    visibility_max_age             = config.param<int>   (k, "visibility_max_age",             5);
    visibility_age_step            = config.param<int>   (k, "visibility_age_step",            1);
    visibility_max_incidence_deg   = config.param<double>(k, "visibility_max_incidence_deg",   75.0);
    visibility_min_filter          = config.param<int>   (k, "visibility_min_filter",          1);
    visibility_keep_frac           = config.param<double>(k, "visibility_keep_frac",           2.0);
    visibility_keep_min_points     = config.param<int>   (k, "visibility_keep_min_points",     2);
    evidence_enabled               = config.param<bool>  (k, "evidence_enabled",               false);
    evidence_resolution            = config.param<double>(k, "evidence_resolution",            0.4);
    evidence_decay                 = config.param<double>(k, "evidence_decay",                 0.7);
    evidence_gain                  = config.param<double>(k, "evidence_gain",                  1.0);
    evidence_on                    = config.param<double>(k, "evidence_on",                    1.5);
    evidence_off                   = config.param<double>(k, "evidence_off",                   0.5);

    // History-suppression depth stays at the configured frame_num_memory; the stored
    // history may be longer to serve the long baseline and the free-space test.
    history_frames = frame_num_memory;
    // Frame history must hold at least the long-baseline frame.
    if (frame_num_memory < compare_baseline_frames)
        frame_num_memory = compare_baseline_frames;
    if (visibility_enabled && frame_num_memory < visibility_max_age)
        frame_num_memory = visibility_max_age;

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
        abs_pose_history_.push_back(lidar_pose());
        last_pose_ = lidar_pose();
        last_dynamic_frame_ = nullptr;
        result.static_frame  = source_frame;
        result.dynamic_frame = nullptr;
        return result;
    }

    // getPose() returns T_world_sensor. Points of the current scan are expressed in the
    // current sensor frame, so the transform that maps them into the previous sensor
    // frame is T_prev^-1 * T_cur (NOT T_cur * T_prev^-1, which is a world-frame delta
    // and only coincides with the correct one for small rotations / poses near the origin).
    // The Kalman pose is T_world_imu; it is converted to T_world_lidar with the extrinsic.
    Eigen::Isometry3d cur_pose     = lidar_pose();
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
    compute_ground_heights(*wf_result.voxelmap);
    if (params_.visibility_enabled) compute_visibility(*wf_result.voxelmap, cur_pose);
    else vis_frac_.clear();
    score_voxels(*wf_result.voxelmap, *voxelmap_history_.back(), cluster_bboxes, T_delta_pose);
    auto t1 = std::chrono::steady_clock::now();
    spdlog::debug("[dynamic_rejection] score_voxels took {:.1f} ms",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    const int nvox = nvox_of(*wf_result.voxelmap);
    propagate_to_neighbors(*wf_result.voxelmap, nvox);
    propagate_to_clusters(*wf_result.voxelmap, cluster_bboxes, historical_bboxes);
    if (params_.evidence_enabled) apply_world_evidence(*wf_result.voxelmap, cur_pose);
    if (params_.point_refine_enabled) refine_points(*wf_result.voxelmap);
    else point_dyn_.clear();

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
    abs_pose_history_.push_back(cur_pose);
    while (static_cast<int>(voxelmap_history_.size()) > params_.frame_num_memory) {
        voxelmap_history_.pop_front();
        pose_history_.pop_front();
        abs_pose_history_.pop_front();
        if (!range_img_history_.empty()) range_img_history_.pop_front();
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
            if (b.contains_parts(mean)) {
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
    const int hist_depth = std::min(params_.history_frames, hist_size) - 1;

    // Baseline frame t-K lives at voxelmap_history_[hist_size - K]
    // (voxelmap_history_.back() is t-1). Centroid noise between two scans is ~constant
    // w.r.t. the time gap (~6-9 cm on indoor data), while a moving object's
    // displacement grows linearly, so comparing against t-K raises the SNR ~K times.
    const int K = std::max(1, std::min(params_.compare_baseline_frames, hist_size));
    const auto& base_map = *voxelmap_history_[hist_size - K];
    // Ego-motion between the baseline scan and now (for the pose-uncertainty term).
    const Eigen::Isometry3d T_base_now = abs_pose_history_[hist_size - K].inverse() * last_pose_;
    const double base_trans = T_base_now.translation().norm();
    const double base_rot   = Eigen::AngleAxisd(T_base_now.linear()).angle();

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
        const bool sparse = cur.num_points < points_thr;
        if (sparse && params_.sparse_voxel_mode == 0) return;
        if (!hag_.empty() && hag_[j] < params_.ground_band_m) return;      // near-ground band
        if (sparse && params_.sparse_voxel_mode == 2) {
            // Too few points for a reliable centroid: decide on the per-point free-space test only.
            const double vis = vis_frac_.empty() ? 0.0 : vis_frac_[j];
            cur.dynamic_score = params_.w_visibility * vis;
            const bool in_dyn = [&] { for (const auto& e : voxel_bboxes_[j]) if (cluster_bboxes[e.first].is_dynamic_bbox()) return true; return false; }();
            const double thr = params_.dynamic_score_threshold * motion_scale_ * (in_dyn ? params_.tier1_threshold_factor : params_.unconstrained_threshold_factor);
            if (vis > 0.0 && cur.dynamic_score > thr) { cur.is_dynamic = true; mark_dynamic(j, cur.mean); }
            return;
        }

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

        // ---- Visibility: fraction of points that appeared in previously free space ----
        const double vis = vis_frac_.empty() ? 0.0 : vis_frac_[j];
        cur.dynamic_score += params_.w_visibility * vis;

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
            // search window -> dynamic. Otherwise decide on the visibility evidence
            // alone (typical for sparse far objects without a voxel match).
            const double base_thr = params_.dynamic_score_threshold * motion_scale_;
            const double thr = was_recently_dynamic ? base_thr * params_.memory_threshold_factor
                                                    : base_thr * params_.unconstrained_threshold_factor;
            if (possibly_dynamic) {
                cur.is_dynamic    = true;
                cur.dynamic_score = params_.dynamic_score_threshold + params_.unmatched_dynamic_margin;
                mark_dynamic(j, cur.mean);
            } else if (vis > 0.0 && cur.dynamic_score > thr) {
                cur.is_dynamic = true;
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
        // Measurement noise (range dependent) + pose uncertainty accumulated over the baseline:
        // thresholds follow the expected error, not the vehicle speed (legacy motion_scale).
        const double sigma = std::min(params_.noise_sigma_max, params_.noise_sigma0 + params_.noise_sigma_slope * range)
                           + params_.pose_err_trans_k * base_trans + params_.pose_err_rot_k * base_rot * range;
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
// compute_ground_heights()
// ===========================================================================
// Local ground = lowest voxel centroid in a ground_cell_m XY cell and its 8 neighbours.

void DynamicObjectRejectionCPU::compute_ground_heights(const gtsam_points::DynamicVoxelMapCPU& voxelmap)
{
    const int nvox = nvox_of(voxelmap);
    hag_.assign(nvox, std::numeric_limits<float>::infinity());
    if (params_.ground_band_m <= 0.0) { hag_.clear(); return; }
    const double inv = 1.0 / params_.ground_cell_m;
    const auto key = [](int x, int y) { return (static_cast<int64_t>(x) << 32) ^ static_cast<uint32_t>(y); };
    std::unordered_map<int64_t, float> zmin;
    std::vector<std::pair<int,int>> cell(nvox);
    for (int j = 0; j < nvox; ++j) {
        const auto& m = voxelmap.lookup_voxel(j).mean;
        cell[j] = {static_cast<int>(std::floor(m.x() * inv)), static_cast<int>(std::floor(m.y() * inv))};
        auto it = zmin.find(key(cell[j].first, cell[j].second));
        if (it == zmin.end()) zmin.emplace(key(cell[j].first, cell[j].second), static_cast<float>(m.z()));
        else it->second = std::min(it->second, static_cast<float>(m.z()));
    }
    for (int j = 0; j < nvox; ++j) {
        float g = std::numeric_limits<float>::infinity();
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) {
            auto it = zmin.find(key(cell[j].first + dx, cell[j].second + dy));
            if (it != zmin.end()) g = std::min(g, it->second);
        }
        hag_[j] = static_cast<float>(voxelmap.lookup_voxel(j).mean.z()) - g;
    }
}


// ===========================================================================
// compute_visibility()
// ===========================================================================
// Free-space test in the PAST sensor frames (no parallax): for each past scan
// t-max_age..t-min_age a spherical range image is built from its own points
// (3x3 min filter, i.e. conservative). Each current point is moved into that past
// sensor frame; if the past ray in its direction reached clearly beyond it, the
// point now occupies space that was observed free -> "appeared" (dynamic evidence).
// Only this direction is used: "farther than before" can be mere disocclusion.

static std::vector<float> build_range_image(const gtsam_points::DynamicVoxelMapCPU& vm, int rows, int cols, double res, int F) {
    std::vector<float> img(rows * cols, std::numeric_limits<float>::infinity());
    const int n = static_cast<int>(vm.gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    for (int v = 0; v < n; ++v)
        for (const auto& pt : vm.lookup_voxel(v).voxel_points) {
            const Eigen::Vector3d q = pt.head<3>();
            const int c = std::min(cols - 1, static_cast<int>((std::atan2(q.y(), q.x()) + M_PI) / res));
            const int r = std::min(rows - 1, std::max(0, static_cast<int>((std::atan2(q.z(), q.head<2>().norm()) + M_PI / 2.0) / res)));
            float& cell = img[r * cols + c];
            cell = std::min(cell, static_cast<float>(q.norm()));
        }
    std::vector<float> out(rows * cols, std::numeric_limits<float>::infinity());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float m = std::numeric_limits<float>::infinity();
            for (int dr = -F; dr <= F; ++dr) {
                const int rr = r + dr; if (rr < 0 || rr >= rows) continue;
                for (int dc = -F; dc <= F; ++dc) m = std::min(m, img[rr * cols + (c + dc + cols) % cols]);
            }
            out[r * cols + c] = m;
        }
    return out;
}

void DynamicObjectRejectionCPU::compute_visibility(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor)
{
    const int nvox = nvox_of(voxelmap);
    vis_frac_.assign(nvox, 0.f);
    vis_pt_.assign(nvox, {});

    const double res = params_.visibility_res_deg * M_PI / 180.0;
    const int cols = static_cast<int>(std::ceil(2.0 * M_PI / res));
    const int rows = static_cast<int>(std::ceil(M_PI / res));

    // Range images are cached per history entry (built once, in their own frame).
    while (range_img_history_.size() < voxelmap_history_.size())
        range_img_history_.push_back(build_range_image(*voxelmap_history_[range_img_history_.size()], rows, cols, res, std::max(0, params_.visibility_min_filter)));

    const int hist = static_cast<int>(voxelmap_history_.size());
    std::vector<int> ages;
    for (int age = params_.visibility_min_age; age <= params_.visibility_max_age; age += std::max(1, params_.visibility_age_step))
        if (hist - age >= 0) ages.push_back(hist - age);
    if (ages.empty()) return;

    std::vector<Eigen::Isometry3d> T_past_now;
    for (int idx : ages) T_past_now.push_back(abs_pose_history_[idx].inverse() * T_world_sensor);

    const double cos_max_inc = std::cos(params_.visibility_max_incidence_deg * M_PI / 180.0);
    parallel_for_voxels(nvox, params_.num_threads, [&](int j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (v.is_wall || v.is_ground || v.is_outlier || v.voxel_points.empty()) return;
        if (v.voxel_points.size() >= 5) {                                  // grazing-incidence filter
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(v.cov.topLeftCorner<3, 3>());
            if (es.info() == Eigen::Success) {
                const Eigen::Vector3d ev = es.eigenvalues();
                if (ev(0) < 0.1 * ev(1) &&
                    std::abs(es.eigenvectors().col(0).dot(v.mean.head<3>().normalized())) < cos_max_inc) return;
            }
        }
        int hits = 0;
        auto& st = vis_pt_[j];
        st.assign(v.voxel_points.size(), 0);
        for (size_t pi = 0; pi < v.voxel_points.size(); ++pi) {
            const auto& pt = v.voxel_points[pi];
            int votes = 0, seen = 0, cons = 0;
            for (size_t a = 0; a < ages.size(); ++a) {
                const Eigen::Vector3d q = T_past_now[a] * pt.head<3>();     // point in past sensor frame
                const int c = std::min(cols - 1, static_cast<int>((std::atan2(q.y(), q.x()) + M_PI) / res));
                const int r = std::min(rows - 1, std::max(0, static_cast<int>((std::atan2(q.z(), q.head<2>().norm()) + M_PI / 2.0) / res)));
                const float past = range_img_history_[ages[a]][r * cols + c];
                if (!std::isfinite(past)) continue;                            // direction not observed
                ++seen;
                const double d = q.norm();
                const double thr = std::max(params_.visibility_abs_thr, params_.visibility_rel_thr * d);
                if (d < past - thr) ++votes;
                else if (std::abs(d - past) <= thr) ++cons;                    // a surface was already there
            }
            if (seen > 0 && 2 * votes > seen) { ++hits; st[pi] = 1; }       // majority of past scans agree
            else if (seen > 0 && 2 * cons > seen) st[pi] = 2;
        }
        vis_frac_[j] = static_cast<float>(hits) / v.voxel_points.size();
    });
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
    std::vector<double> vis_sum   (n_clusters, 0.0);

    // Dynamic-wins multi-cluster assignment: a voxel contributes to ALL base bboxes containing it.
    for (int j = 0; j < nvox; ++j) {
        const bool dyn = voxelmap.lookup_voxel(j).is_dynamic;
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            if (!in_base) continue;
            ++total_count[c];
            if (dyn) ++dynamic_count[c];
            if (!vis_frac_.empty()) vis_sum[c] += vis_frac_[j];
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

        // Object-level plausibility: tall enough and coherently moving in the world frame.
        const double vis_mean = vis_sum[c] / total_count[c];
        const double speed    = std::max(cluster_bboxes[c].get_speed_xy(), cluster_bboxes[c].get_window_speed());
        const bool tall   = cluster_bboxes[c].get_size().z() >= params_.min_obj_height;
        const bool moving = params_.min_obj_speed <= 0.0 || speed >= params_.min_obj_speed || vis_mean >= params_.moving_vis_alt;
        const bool is_dyn = ratio > eff_prop_threshold && tall && moving;
        // Fast confirmation: very strong evidence in this frame -> remove now, without waiting
        // for the hysteresis (the tracker still counts it as one confirmed frame).
        if (is_dyn && params_.fast_confirm_ratio_factor > 0.0 &&
            ratio > params_.fast_confirm_ratio_factor * eff_prop_threshold && vis_mean >= params_.fast_confirm_min_vis)
            was_dynamic[c] = true;
        cluster_bboxes[c].set_dynamic(is_dyn);  // feedback for update_dynamic_feedback()
        spdlog::debug("[dynamic_rejection] cluster {}: {}/{} dynamic ({:.1f}%) threshold={:.2f} -> {} (was {})",
                      c, dynamic_count[c], total_count[c], ratio * 100.0, eff_prop_threshold,
                      is_dyn ? "DYNAMIC" : "static", was_dynamic[c] ? "DYNAMIC" : "static");
    }

    confirmed_bboxes_.clear();
    for (int c = 0; c < n_clusters; ++c) if (was_dynamic[c]) confirmed_bboxes_.push_back(cluster_bboxes[c]);

    // Final voxel assignment.
    const auto assign = [&](bool strict) {
        parallel_for_voxels(nvox, params_.num_threads, [&](int j) {
            auto& v = voxelmap.lookup_voxel(j);
            const bool own = v.is_dynamic || (!vis_frac_.empty() && vis_frac_[j] > 0.f);  // per-voxel evidence (pre-assignment)
            const bool near_ground = !hag_.empty() && hag_[j] < params_.ground_band_m;
            // Feet / wheels: low voxels inside the XY footprint of a confirmed tall dynamic bbox,
            // slightly above the local ground (the ground itself stays static).
            if ((v.is_ground || near_ground) && !v.is_wall && params_.foot_min_hag >= 0.0 && !hag_.empty() &&
                hag_[j] >= params_.foot_min_hag) {
                for (int c = 0; c < n_clusters; ++c) {
                    if (!was_dynamic[c]) continue;
                    const auto& b = cluster_bboxes[c];
                    if (b.get_size().z() < params_.min_obj_height) continue;
                    const Eigen::Vector3d d = v.mean.head<3>() - b.get_center();
                    const Eigen::Vector3d hs = 0.5 * b.get_size();
                    if (std::abs(d.x()) <= hs.x() && std::abs(d.y()) <= hs.y() &&
                        d.z() <= hs.z() && d.z() >= -hs.z() - params_.ground_band_m) { v.is_dynamic = true; return; }
                }
            }
            // Structural voxels are never removed (ground under a person must stay in the map).
            if (v.is_wall || v.is_ground || v.is_outlier) { v.is_dynamic = false; return; }

            bool in_any_base = false, in_dyn_base = false, in_dyn_inflated = false;
            for (const auto& [c, in_base] : voxel_bboxes_[j]) {
                if (!was_dynamic[c]) { if (in_base) in_any_base = true; continue; }
                if (near_ground && cluster_bboxes[c].get_size().z() < params_.min_obj_height) continue;
                if (in_base) in_dyn_base = true; else in_dyn_inflated = true;
            }
            if (in_dyn_base) { v.is_dynamic = true; return; }
            if (strict || near_ground) { v.is_dynamic = false; return; }   // anomalous frame / ground band: confirmed objects only
            const bool need = params_.inflated_requires_evidence;
            if (in_dyn_inflated) { v.is_dynamic = !need || own; return; }
            if (in_any_base) { v.is_dynamic = false; return; }             // inside a static cluster only

            // Outside all bboxes: historical bboxes of confirmed-dynamic tracks.
            for (const auto& hbbox : historical_bboxes) {
                if (hbbox.contains(v.mean) || hbbox.contains_inflated(v.mean, inflate_params_)) {
                    v.is_dynamic = !need || own;
                    return;
                }
            }
            const bool vis_keep = !vis_frac_.empty() && vis_frac_[j] >= params_.visibility_keep_frac &&
                                  static_cast<int>(v.voxel_points.size()) >= params_.visibility_keep_min_points;
            v.is_dynamic = (params_.keep_voxel_evidence && v.is_dynamic) || vis_keep;
        });
    };
    std::vector<char> pre(nvox);
    for (int j = 0; j < nvox; ++j) pre[j] = voxelmap.lookup_voxel(j).is_dynamic;
    assign(false);

    // Anomalous frame guard: an implausibly large dynamic fraction usually means a pose /
    // deskew problem, not a crowd. Fall back to confirmed-dynamic objects only.
    size_t dyn_pts = 0, all_pts = 0;
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        all_pts += v.voxel_points.size(); if (v.is_dynamic) dyn_pts += v.voxel_points.size();
    }
    if (all_pts > 0 && static_cast<double>(dyn_pts) / all_pts > params_.frame_max_dynamic_frac) {
        spdlog::debug("[dynamic_rejection] anomalous frame: {:.1f}% dynamic -> confirmed objects only", 100.0 * dyn_pts / all_pts);
        for (int j = 0; j < nvox; ++j) voxelmap.lookup_voxel(j).is_dynamic = pre[j];
        assign(true);
    }
}


// ===========================================================================
// refine_points()
// ===========================================================================
// Point-level labelling on top of the voxel decision:
//  1. points of dynamic voxels are dynamic, except those whose past rays saw a surface at
//     the same range (static veto) -> removes halos in mixed boundary voxels;
//  2. region growing from the dynamic points through a KD-tree, restricted to confirmed
//     dynamic bboxes (+margin), non-structural, above the ground band and not vetoed
//     -> recovers borders and fragments that the clustering missed.

void DynamicObjectRejectionCPU::refine_points(const gtsam_points::DynamicVoxelMapCPU& voxelmap)
{
    const int nvox = nvox_of(voxelmap);
    point_dyn_.assign(nvox, {});
    std::vector<Eigen::Vector4d> pts;
    std::vector<std::pair<int, int>> owner;
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        point_dyn_[j].assign(v.voxel_points.size(), 0);
        for (size_t k = 0; k < v.voxel_points.size(); ++k) { pts.push_back(v.voxel_points[k]); owner.emplace_back(j, static_cast<int>(k)); }
    }
    if (pts.empty()) return;
    const auto vis_state = [&](int j, int k) -> uint8_t {
        return (static_cast<int>(vis_pt_.size()) == nvox && !vis_pt_[j].empty()) ? vis_pt_[j][k] : 0;
    };

    std::vector<uint8_t> lab(pts.size(), 0);
    std::vector<int> queue;
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto [j, k] = owner[i];
        if (!voxelmap.lookup_voxel(j).is_dynamic) continue;
        if (params_.point_static_veto && vis_state(j, k) == 2) continue;
        lab[i] = 1; queue.push_back(static_cast<int>(i));
    }

    if ((!confirmed_bboxes_.empty() || params_.point_grow_on_visibility) && !queue.empty()) {
        const auto allowed = [&](int i) {
            const auto [j, k] = owner[i];
            const auto& v = voxelmap.lookup_voxel(j);
            if (v.is_wall || v.is_ground || v.is_outlier) return false;
            if (!hag_.empty() && hag_[j] < params_.ground_band_m) return false;
            if (params_.point_static_veto && vis_state(j, k) == 2) return false;
            const Eigen::Vector3d p = pts[i].head<3>();
            for (const auto& b : confirmed_bboxes_) {
                const Eigen::Vector3d d = (p - b.get_center()).cwiseAbs() - 0.5 * b.get_size();
                if ((d.array() <= params_.point_bbox_margin).all()) return true;
            }
            return params_.point_grow_on_visibility && vis_state(j, k) == 1;
        };
        gtsam_points::KdTree tree(pts.data(), static_cast<int>(pts.size()));
        const int kk = std::max(1, params_.point_knn);
        std::vector<size_t> idx(kk); std::vector<double> sq(kk);
        for (size_t q = 0; q < queue.size(); ++q) {
            const int i = queue[q];
            const double r = std::min(params_.point_grow_radius_max,
                                      params_.point_grow_radius0 + params_.point_grow_radius_k * pts[i].head<3>().norm());
            const size_t n = tree.knn_search(pts[i].data(), kk, idx.data(), sq.data());
            for (size_t m = 0; m < n; ++m) {
                const int nb = static_cast<int>(idx[m]);
                if (lab[nb] || sq[m] > r * r || !allowed(nb)) continue;
                lab[nb] = 1; queue.push_back(nb);
            }
        }
    }
    for (size_t i = 0; i < pts.size(); ++i) point_dyn_[owner[i].first][owner[i].second] = lab[i];
}


// ===========================================================================
// make_hybrid_frame()
// ===========================================================================

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::make_hybrid_frame(
    const PreprocessedFrame::Ptr& frame, const DynamicObjectRejectionParamsCPU& params)
{
    if (!params.hybrid_enabled || !frame || !frame->raw_points || frame->raw_points->points.empty()) return frame;
    const auto& raw = *frame->raw_points;
    const double r0 = params.hybrid_min_range * params.hybrid_min_range;
    const double r1 = params.hybrid_max_range * params.hybrid_max_range;
    std::vector<int> sel;
    for (int i = 0; i < static_cast<int>(raw.points.size()); ++i) {
        const double d2 = raw.points[i].head<2>().squaredNorm();
        if (d2 >= r0 && d2 < r1 && raw.points[i].allFinite()) sel.push_back(i);
    }
    if (sel.empty()) return frame;
    const int cap = params.hybrid_max_points;
    const double step = (cap > 0 && static_cast<int>(sel.size()) > cap) ? static_cast<double>(sel.size()) / cap : 1.0;

    auto aug = std::make_shared<PreprocessedFrame>(*frame);
    aug->neighbors.clear();
    aug->k_neighbors = 0;
    const bool has_int = !aug->intensities.empty();
    for (double f = 0.0; f < sel.size(); f += step) {
        const int i = sel[static_cast<int>(f)];
        Eigen::Vector4d p = raw.points[i]; p.w() = 1.0;
        aug->points.push_back(p);
        aug->times.push_back(std::numeric_limits<double>::quiet_NaN());   // marker: extra point
        if (has_int) aug->intensities.push_back(i < static_cast<int>(raw.intensities.size()) ? raw.intensities[i] : 0.0);
    }
    return aug;
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

    const bool per_point = static_cast<int>(point_dyn_.size()) == nvox;
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (per_point && !point_dyn_[j].empty()) {
            for (size_t k = 0; k < v.voxel_points.size(); ++k) {
                if (k < v.voxel_times.size() && std::isnan(v.voxel_times[k])) continue;   // hybrid extra point
                const bool d = point_dyn_[j][k];
                (d ? dynamic_pts : static_pts).push_back(v.voxel_points[k]);
                if (k < v.voxel_intensities.size()) (d ? dynamic_int : static_int).push_back(v.voxel_intensities[k]);
                if (k < v.voxel_times.size())       (d ? dynamic_tim : static_tim).push_back(v.voxel_times[k]);
            }
            continue;
        }

        auto& pts  = v.is_dynamic ? dynamic_pts : static_pts;
        auto& ints = v.is_dynamic ? dynamic_int : static_int;
        auto& tims = v.is_dynamic ? dynamic_tim : static_tim;

        bool has_extra = false;
        for (const double t : v.voxel_times) if (std::isnan(t)) { has_extra = true; break; }
        if (has_extra) {   // hybrid extra points never reach the output frames
            for (size_t k = 0; k < v.voxel_points.size(); ++k) {
                if (k < v.voxel_times.size() && std::isnan(v.voxel_times[k])) continue;
                pts.push_back(v.voxel_points[k]);
                if (k < v.voxel_intensities.size()) ints.push_back(v.voxel_intensities[k]);
                if (k < v.voxel_times.size())       tims.push_back(v.voxel_times[k]);
            }
            continue;
        }

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