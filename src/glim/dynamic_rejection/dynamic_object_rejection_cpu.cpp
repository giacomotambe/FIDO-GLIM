#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <unordered_set>
#include <cstring>

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
    point_ground_cut               = config.param<double>(k, "point_ground_cut",               -1.0);
    point_ground_cut_k             = config.param<double>(k, "point_ground_cut_k",             0.0);
    point_ground_cut_min_range     = config.param<double>(k, "point_ground_cut_min_range",     0.0);
    fs_enabled                     = config.param<bool>  (k, "fs_enabled",                     false);
    vis_raw                        = config.param<bool>  (k, "vis_raw",                        false);
    large_box_len                  = config.param<double>(k, "large_box_len",                  1e9);
    fut_delay                      = config.param<int>   (k, "fut_delay",                      0);
    min_obj_vis                    = config.param<double>(k, "min_obj_vis",                    0.0);
    view_change_k                  = config.param<double>(k, "view_change_k",                  0.0);
    min_obj_vis_max_range          = config.param<double>(k, "min_obj_vis_max_range",          1e9);
    fut_output_delayed             = config.param<bool>  (k, "fut_output_delayed",             false);
    fut_min_age                    = config.param<int>   (k, "fut_min_age",                    4);
    fut_age_step                   = config.param<int>   (k, "fut_age_step",                   3);
    fut_ratio                      = config.param<double>(k, "fut_ratio",                      0.6);
    fut_max_range                  = config.param<double>(k, "fut_max_range",                  22.0);
    icp_enable                     = config.param<bool>  (k, "icp_enable",                     false);
    icp_min_len                    = config.param<double>(k, "icp_min_len",                    2.5);
    icp_gap_frames                 = config.param<int>   (k, "icp_gap_frames",                 5);
    icp_min_speed                  = config.param<double>(k, "icp_min_speed",                  0.5);
    icp_max_rms                    = config.param<double>(k, "icp_max_rms",                    0.2);
    icp_min_inlier                 = config.param<double>(k, "icp_min_inlier",                 0.5);
    icp_max_pts                    = config.param<int>   (k, "icp_max_pts",                    300);
    large_box_ratio                = config.param<double>(k, "large_box_ratio",                0.7);
    large_box_vis                  = config.param<double>(k, "large_box_vis",                  0.3);
    vis_raw_max_range              = config.param<double>(k, "vis_raw_max_range",              40.0);
    fs_raw                         = config.param<bool>  (k, "fs_raw",                         false);
    pe_mode                        = config.param<int>   (k, "pe_mode",                        0);
    pe_min_points                  = config.param<int>   (k, "pe_min_points",                  2);
    pe_max_range                   = config.param<double>(k, "pe_max_range",                   20.0);
    split_enable                   = config.param<bool>  (k, "split_enable",                   false);
    split_radius                   = config.param<double>(k, "split_radius",                   0.5);
    split_min_dyn                  = config.param<int>   (k, "split_min_dyn",                  2);
    split_ratio                    = config.param<double>(k, "split_ratio",                    0.6);
    split_min_vis                  = config.param<double>(k, "split_min_vis",                  0.2);
    fs_res                         = config.param<double>(k, "fs_res",                         0.3);
    fs_min_free                    = config.param<int>   (k, "fs_min_free",                    3);
    fs_free_ratio                  = config.param<double>(k, "fs_free_ratio",                  3.0);
    fs_ray_stride                  = config.param<int>   (k, "fs_ray_stride",                  2);
    fs_max_range                   = config.param<double>(k, "fs_max_range",                   30.0);
    fs_margin                      = config.param<double>(k, "fs_margin",                      0.3);
    fs_margin_rel                  = config.param<double>(k, "fs_margin_rel",                  0.03);
    fs_dilate                      = config.param<int>   (k, "fs_dilate",                      1);
    fs_occ_static_only             = config.param<bool>  (k, "fs_occ_static_only",             false);
    fs_decay                       = config.param<double>(k, "fs_decay",                       1.0);
    fs_flag_nb                     = config.param<bool>  (k, "fs_flag_nb",                     false);
    fs_dda                         = config.param<bool>  (k, "fs_dda",                         false);
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
    if (fut_delay > 0 && frame_num_memory < fut_delay + 1)
        frame_num_memory = fut_delay + 1;

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
        if (params_.vis_raw && source_frame && source_frame->raw_points && !source_frame->raw_points->points.empty()) push_raw_range_image(source_frame);
        pose_history_.push_back(Eigen::Isometry3d::Identity());  // no predecessor: identity delta
        abs_pose_history_.push_back(lidar_pose());
        last_pose_ = lidar_pose();
        last_dynamic_frame_ = nullptr;
        if (params_.fs_enabled) { if (params_.fs_raw && source_frame && source_frame->raw_points) fs_integrate_points(source_frame->raw_points->points, lidar_pose()); else fs_integrate(*wf_result.voxelmap, lidar_pose()); }
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
    if (params_.fs_enabled) fs_query(*wf_result.voxelmap, cur_pose);
    score_voxels(*wf_result.voxelmap, *voxelmap_history_.back(), cluster_bboxes, T_delta_pose);
    auto t1 = std::chrono::steady_clock::now();
    spdlog::debug("[dynamic_rejection] score_voxels took {:.1f} ms",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    const int nvox = nvox_of(*wf_result.voxelmap);
    propagate_to_neighbors(*wf_result.voxelmap, nvox);
    cur_pose_ = cur_pose; cur_stamp_ = source_frame ? source_frame->stamp : cur_stamp_ + 0.1; ++icp_frame_;
    propagate_to_clusters(*wf_result.voxelmap, cluster_bboxes, historical_bboxes);
    if (params_.evidence_enabled) apply_world_evidence(*wf_result.voxelmap, cur_pose);
    if (params_.point_refine_enabled) refine_points(*wf_result.voxelmap);
    else {
        point_dyn_.clear();
        if (params_.point_ground_cut > 0.0) split_ground_points(*wf_result.voxelmap);
        if (params_.pe_mode > 0) point_evidence_labels(*wf_result.voxelmap);
    }

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
    if (params_.fs_enabled) { if (params_.fs_raw && source_frame && source_frame->raw_points) fs_integrate_points(source_frame->raw_points->points, cur_pose); else fs_integrate(*wf_result.voxelmap, cur_pose); }
    voxelmap_history_.push_back(wf_result.voxelmap);
    if (params_.vis_raw && source_frame && source_frame->raw_points && !source_frame->raw_points->points.empty()) push_raw_range_image(source_frame);
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

    if (params_.fut_delay > 0) {
        Pending p; p.static_frame = result.static_frame; p.dynamic_frame = result.dynamic_frame;
        p.stamp = source_frame ? source_frame->stamp : 0.0; p.pose = cur_pose;
        collect_delay_candidates(*wf_result.voxelmap, cluster_bboxes, p);
        pending_.push_back(std::move(p));
        if (static_cast<int>(pending_.size()) > params_.fut_delay) finalize_delayed(result);
    }
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
                           + params_.pose_err_trans_k * base_trans + params_.pose_err_rot_k * base_rot * range
                           // Viewpoint change: the sampled part of a surface moves with the parallax angle
                           // (~ baseline / range), up to about a voxel.
                           + params_.view_change_k * voxel_res * std::min(1.0, base_trans / std::max(range, 1.0));
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

static std::vector<float> build_range_image_pts(const std::vector<Eigen::Vector4d>& pts, int rows, int cols, double res, int F, double rmax) {
    std::vector<float> img(rows * cols, std::numeric_limits<float>::infinity());
    const double r2 = rmax * rmax;
    for (const auto& pt : pts) {
        const Eigen::Vector3d q = pt.head<3>();
        const double d2 = q.squaredNorm();
        if (!(d2 > 1e-4) || d2 > r2) continue;
        const int c = std::min(cols - 1, static_cast<int>((std::atan2(q.y(), q.x()) + M_PI) / res));
        const int r = std::min(rows - 1, std::max(0, static_cast<int>((std::atan2(q.z(), q.head<2>().norm()) + M_PI / 2.0) / res)));
        float& cell = img[r * cols + c];
        cell = std::min(cell, static_cast<float>(std::sqrt(d2)));
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

void DynamicObjectRejectionCPU::push_raw_range_image(const PreprocessedFrame::Ptr& frame) {
    // Called right after the frame is pushed into voxelmap_history_: the lazy builder in
    // compute_visibility() then finds its image already there.
    const double res = params_.visibility_res_deg * M_PI / 180.0;
    const int cols = static_cast<int>(std::ceil(2.0 * M_PI / res)), rows = static_cast<int>(std::ceil(M_PI / res));
    while (range_img_history_.size() + 1 < voxelmap_history_.size())
        range_img_history_.push_back(build_range_image(*voxelmap_history_[range_img_history_.size()], rows, cols, res, std::max(0, params_.visibility_min_filter)));
    if (range_img_history_.size() + 1 == voxelmap_history_.size())
        range_img_history_.push_back(build_range_image_pts(frame->raw_points->points, rows, cols, res, std::max(0, params_.visibility_min_filter), params_.vis_raw_max_range));
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
// Free-space map (fs_query / fs_integrate)
// ===========================================================================

static inline int64_t fs_pack(int64_t x, int64_t y, int64_t z) {
    return ((x & 0x1FFFFF) << 42) | ((y & 0x1FFFFF) << 21) | (z & 0x1FFFFF);
}
static inline int64_t fs_key(const Eigen::Vector3d& p, double inv) {
    return fs_pack(static_cast<int64_t>(std::floor(p.x() * inv)), static_cast<int64_t>(std::floor(p.y() * inv)),
                   static_cast<int64_t>(std::floor(p.z() * inv)));
}

float DynamicObjectRejectionCPU::fs_decayed(float v, uint32_t last) const {
    if (params_.fs_decay >= 1.0 || last == 0) return v;
    const uint32_t k = fs_frame_ - last;
    return k < fs_pow_.size() ? v * fs_pow_[k] : 0.f;
}

void DynamicObjectRejectionCPU::fs_query(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T)
{
    const int nvox = nvox_of(voxelmap);
    if (static_cast<int>(vis_frac_.size()) != nvox) vis_frac_.assign(nvox, 0.f);
    const double res = params_.fs_res, inv = 1.0 / res, r2max = params_.fs_max_range * params_.fs_max_range;
    fs_pt_.assign(nvox, {});
    parallel_for_voxels(nvox, params_.num_threads, [&](int j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (v.is_wall || v.is_ground || v.is_outlier || v.voxel_points.empty()) return;
        if (v.mean.head<2>().squaredNorm() > r2max) return;
        int hits = 0;
        auto& fp = fs_pt_[j]; fp.assign(v.voxel_points.size(), 0);
        for (size_t pk = 0; pk < v.voxel_points.size(); ++pk) {
            const auto& p = v.voxel_points[pk];
            const Eigen::Vector3d w = T * p.head<3>();
            auto it = fs_map_.find(fs_key(w, inv));
            if (it == fs_map_.end()) continue;
            const auto& c = it->second;
            const float fr = fs_decayed(c.free, c.last_free), oc = fs_decayed(c.occ, c.last_occ);
            // Known free (enough free observations, rarely occupied) ...
            if (fr < params_.fs_min_free || fr < params_.fs_free_ratio * oc) continue;
            // ... and a void neighbourhood (DUFOMap-style dilation).
            if (params_.fs_flag_nb) { if (c.near == 0) { ++hits; fp[pk] = 1; } continue; }
            bool void_nb = true;
            for (int dx = -params_.fs_dilate; dx <= params_.fs_dilate && void_nb; ++dx)
                for (int dy = -params_.fs_dilate; dy <= params_.fs_dilate && void_nb; ++dy)
                    for (int dz = -params_.fs_dilate; dz <= params_.fs_dilate; ++dz) {
                        if (!dx && !dy && !dz) continue;
                        auto jt = fs_map_.find(fs_key(w + res * Eigen::Vector3d(dx, dy, dz), inv));
                        if (jt != fs_map_.end() &&
                            fs_decayed(jt->second.occ, jt->second.last_occ) * params_.fs_free_ratio > fs_decayed(jt->second.free, jt->second.last_free)) { void_nb = false; break; }
                    }
            if (void_nb) { ++hits; fp[pk] = 1; }
        }
        vis_frac_[j] = std::max(vis_frac_[j], static_cast<float>(hits) / v.voxel_points.size());
    });
}

void DynamicObjectRejectionCPU::fs_integrate_points(const std::vector<Eigen::Vector4d>& pts, const Eigen::Isometry3d& T)
{
    // Same as fs_integrate() on the raw scan (denser rays; labels unknown, so every endpoint is occupied).
    ++fs_frame_;
    if (params_.fs_decay < 1.0 && fs_pow_.empty()) {
        fs_pow_.resize(2000); double q = 1.0;
        for (auto& x : fs_pow_) { x = static_cast<float>(q); q *= params_.fs_decay; }
    }
    const double res = params_.fs_res, inv = 1.0 / res, r2max = params_.fs_max_range * params_.fs_max_range;
    const Eigen::Vector3d o = T.translation();
    const int stride = std::max(1, params_.fs_ray_stride);
    const auto bump = [&](float& v, uint32_t& last) {
        if (last == fs_frame_) return;
        v = fs_decayed(v, last) + 1.f; if (v > 60000.f) v = 60000.f; last = fs_frame_;
    };
    for (size_t i = 0; i < pts.size(); i += stride) {
        const auto& p = pts[i];
        const double d2 = p.head<3>().squaredNorm();
        if (!(d2 > 1e-4) || d2 > r2max) continue;
        const Eigen::Vector3d w = T * p.head<3>();
        { auto& ce = fs_map_[fs_key(w, inv)]; bump(ce.occ, ce.last_occ); ce.marked = true; }
        const double d = std::sqrt(d2);
        const double L = d - std::max(params_.fs_margin, params_.fs_margin_rel * d);
        if (L <= 0.0) continue;
        const Eigen::Vector3d dir = (w - o) / d;
        int64_t prev = -1;
        for (double t = 0.5 * res; t < L; t += 0.5 * res) {
            const int64_t k = fs_key(o + t * dir, inv);
            if (k == prev) continue;
            prev = k; auto& cf = fs_map_[k]; bump(cf.free, cf.last_free);
        }
    }
    if (fs_frame_ % 20 == 0) {
        const double keep2 = std::pow(params_.fs_max_range + 10.0, 2);
        auto sx = [](int64_t a) { return (a & 0x100000) ? a - 0x200000 : a; };
        for (auto it = fs_map_.begin(); it != fs_map_.end();) {
            const int64_t k = it->first;
            const Eigen::Vector2d cxy((sx((k >> 42) & 0x1FFFFF) + 0.5) * res, (sx((k >> 21) & 0x1FFFFF) + 0.5) * res);
            if ((cxy - o.head<2>()).squaredNorm() > keep2) it = fs_map_.erase(it); else ++it;
        }
    }
}

void DynamicObjectRejectionCPU::fs_integrate(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T)
{
    ++fs_frame_;
    if (params_.fs_decay < 1.0 && fs_pow_.empty()) {
        fs_pow_.resize(2000); double q = 1.0;
        for (auto& x : fs_pow_) { x = static_cast<float>(q); q *= params_.fs_decay; }
    }
    const double res = params_.fs_res, inv = 1.0 / res, r2max = params_.fs_max_range * params_.fs_max_range;
    const Eigen::Vector3d o = T.translation();
    const int stride = std::max(1, params_.fs_ray_stride);
    const int nvox = nvox_of(voxelmap);
    const bool per_point = static_cast<int>(point_dyn_.size()) == nvox;
    const auto bump = [&](float& v, uint32_t& last) {
        if (last == fs_frame_) return false;
        v = fs_decayed(v, last) + 1.f; if (v > 60000.f) v = 60000.f; last = fs_frame_; return true;
    };
    const auto mark_free = [&](int64_t k) { auto& cf = fs_map_[k]; bump(cf.free, cf.last_free); };
    int cnt = 0;
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        for (size_t pi = 0; pi < v.voxel_points.size(); ++pi) {
            if (++cnt % stride) continue;
            const auto& p = v.voxel_points[pi];
            const double d2 = p.head<3>().squaredNorm();
            if (d2 > r2max || d2 < 1e-4) continue;
            const Eigen::Vector3d w = T * p.head<3>();
            // Endpoint: occupied (optionally only static points, so moving objects do not
            // keep their paths "occupied").
            const bool dyn = per_point && !point_dyn_[j].empty() ? point_dyn_[j][pi] != 0 : v.is_dynamic;
            if (!(params_.fs_occ_static_only && dyn)) {
                const int64_t ke = fs_key(w, inv);
                auto& ce = fs_map_[ke];
                bump(ce.occ, ce.last_occ);
                if (!ce.marked) {
                    ce.marked = true;
                    if (params_.fs_flag_nb) {
                        const int64_t x = static_cast<int64_t>(std::floor(w.x() * inv)), y = static_cast<int64_t>(std::floor(w.y() * inv)),
                                      z = static_cast<int64_t>(std::floor(w.z() * inv));
                        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz)
                            if (dx || dy || dz) { auto& cn = fs_map_[fs_pack(x + dx, y + dy, z + dz)]; if (cn.near < 60000) ++cn.near; }
                    }
                }
            }
            const double d = std::sqrt(d2);
            const double L = d - std::max(params_.fs_margin, params_.fs_margin_rel * d);
            if (L <= 0.0) continue;
            const Eigen::Vector3d dir = (w - o) / d;
            if (!params_.fs_dda) {
                int64_t prev = -1;
                for (double t = 0.5 * res; t < L; t += 0.5 * res) {
                    const int64_t k = fs_key(o + t * dir, inv);
                    if (k == prev) continue;
                    prev = k; mark_free(k);
                }
                continue;
            }
            // Amanatides-Woo traversal: every crossed cell exactly once.
            Eigen::Vector3i c((o * inv).array().floor().cast<int>());
            const Eigen::Vector3i c_end(((o + L * dir) * inv).array().floor().cast<int>());
            Eigen::Vector3i step; Eigen::Vector3d tmax, tdelta;
            for (int a = 0; a < 3; ++a) {
                if (dir[a] > 1e-12)       { step[a] = 1;  tmax[a] = ((c[a] + 1) * res - o[a]) / dir[a]; tdelta[a] = res / dir[a]; }
                else if (dir[a] < -1e-12) { step[a] = -1; tmax[a] = (c[a] * res - o[a]) / dir[a];       tdelta[a] = -res / dir[a]; }
                else                      { step[a] = 0;  tmax[a] = std::numeric_limits<double>::infinity(); tdelta[a] = tmax[a]; }
            }
            for (int it = 0; it < 4096; ++it) {
                mark_free(fs_pack(c.x(), c.y(), c.z()));
                if (c == c_end) break;
                int a = (tmax.x() < tmax.y()) ? (tmax.x() < tmax.z() ? 0 : 2) : (tmax.y() < tmax.z() ? 1 : 2);
                if (tmax[a] > L) break;
                c[a] += step[a]; tmax[a] += tdelta[a];
            }
        }
    }
    // Keep the map local.
    if (fs_frame_ % 20 == 0) {
        const double keep2 = std::pow(params_.fs_max_range + 10.0, 2);
        auto sx = [](int64_t a) { return (a & 0x100000) ? a - 0x200000 : a; };
        for (auto it = fs_map_.begin(); it != fs_map_.end();) {
            const int64_t k = it->first;
            const Eigen::Vector2d cxy((sx((k >> 42) & 0x1FFFFF) + 0.5) * res, (sx((k >> 21) & 0x1FFFFF) + 0.5) * res);
            if ((cxy - o.head<2>()).squaredNorm() > keep2) it = fs_map_.erase(it); else ++it;
        }
    }
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
    std::vector<std::vector<int>> members((params_.split_enable || params_.icp_enable) ? n_clusters : 0);

    // Dynamic-wins multi-cluster assignment: a voxel contributes to ALL base bboxes containing it.
    for (int j = 0; j < nvox; ++j) {
        const bool dyn = voxelmap.lookup_voxel(j).is_dynamic;
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            if (!in_base) continue;
            ++total_count[c];
            if (dyn) ++dynamic_count[c];
            if (params_.split_enable || params_.icp_enable) members[c].push_back(j);
            if (!vis_frac_.empty()) vis_sum[c] += vis_frac_[j];
        }
    }

    // ICP motion of vehicle-sized clusters: world-frame points vs the same track icp_gap_frames ago.
    std::vector<int> icp_state(n_clusters, -1);   // -1 unknown, 0 static, 1 moving
    if (params_.icp_enable) {
        for (int c = 0; c < n_clusters; ++c) {
            const auto& b = cluster_bboxes[c];
            const int tid = b.get_track_id();
            if (tid < 0 || std::max(b.get_size().x(), b.get_size().y()) <= params_.icp_min_len) continue;
            std::vector<Eigen::Vector3d> w;
            for (int j : members[c]) {
                if (!hag_.empty() && hag_[j] < params_.ground_band_m) continue;
                for (const auto& p : voxelmap.lookup_voxel(j).voxel_points) w.push_back(cur_pose_ * p.head<3>());
            }
            if (w.size() < 10) continue;
            if (static_cast<int>(w.size()) > params_.icp_max_pts) {
                std::vector<Eigen::Vector3d> s; const double st = static_cast<double>(w.size()) / params_.icp_max_pts;
                for (double f = 0; f < w.size(); f += st) s.push_back(w[static_cast<size_t>(f)]);
                w.swap(s);
            }
            auto& h = icp_hist_[tid];
            const IcpEntry* ref = nullptr;
            for (const auto& e : h) if (icp_frame_ - e.frame >= params_.icp_gap_frames) ref = &e;
            if (ref && ref->pts.size() >= 10 && cur_stamp_ > ref->stamp) {
                Eigen::Vector3d cs = Eigen::Vector3d::Zero(), cr = Eigen::Vector3d::Zero();
                for (const auto& p : w) cs += p; cs /= w.size();
                for (const auto& p : ref->pts) cr += p; cr /= ref->pts.size();
                double best_in = -1, best_rms = 1e9; Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
                for (const Eigen::Vector3d& t0 : {Eigen::Vector3d::Zero().eval(), (cr - cs).eval()}) {
                    Eigen::Vector3d t = t0; double rms = 0;
                    const double in = icp_translation(w, ref->pts, t, rms, 0.5);
                    if (in > best_in + 1e-6 || (std::abs(in - best_in) < 1e-6 && rms < best_rms)) { best_in = in; best_rms = rms; best_t = t; }
                }
                const double spd = best_t.head<2>().norm() / (cur_stamp_ - ref->stamp);
                if (best_in >= params_.icp_min_inlier && best_rms <= params_.icp_max_rms)
                    icp_state[c] = spd >= params_.icp_min_speed ? 1 : 0;
            }
            h.push_back({icp_frame_, cur_stamp_, std::move(w)});
            while (h.size() > static_cast<size_t>(params_.icp_gap_frames + 3)) h.pop_front();
        }
        for (auto it = icp_hist_.begin(); it != icp_hist_.end();)
            if (it->second.empty() || icp_frame_ - it->second.back().frame > 20) it = icp_hist_.erase(it); else ++it;
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
        {
            const double vm = vis_sum[c] / total_count[c];
            const double sp = std::max(cluster_bboxes[c].get_speed_xy(), cluster_bboxes[c].get_window_speed());
            const bool tall_c = cluster_bboxes[c].get_size().z() >= params_.min_obj_height;
            const bool large_c = std::max(cluster_bboxes[c].get_size().x(), cluster_bboxes[c].get_size().y()) > params_.large_box_len;
            bool strong_c = !large_c || (ratio > params_.large_box_ratio && vm >= params_.large_box_vis);
            if (params_.icp_enable && large_c && icp_state[c] >= 0) strong_c = icp_state[c] == 1;
            const bool free_c = vm >= params_.min_obj_vis || cluster_bboxes[c].get_center().head<2>().norm() > params_.min_obj_vis_max_range;
            cluster_bboxes[c].set_frame_evidence(tall_c && strong_c && free_c ? ratio + vm : 0.0,
                                                 dynamic_count[c] == 0 && vm <= 0.0 && sp < params_.min_obj_speed);
        }

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
        const bool seen_free = vis_mean >= params_.min_obj_vis || cluster_bboxes[c].get_center().head<2>().norm() > params_.min_obj_vis_max_range;   // free-space evidence required (parked cars: centroid shift only)
        // Vehicle-sized clusters (longer than large_box_len) need stronger evidence: they are
        // mostly parked cars / hedges, whose voxel noise would otherwise confirm them.
        const bool large  = std::max(cluster_bboxes[c].get_size().x(), cluster_bboxes[c].get_size().y()) > params_.large_box_len;
        bool is_dyn = large ? (ratio > params_.large_box_ratio && vis_mean >= params_.large_box_vis && tall)
                            : (ratio > eff_prop_threshold && tall && moving && seen_free);
        // ICP verdict overrides the voxel ratio for vehicle-sized clusters (when available).
        if (params_.icp_enable && large && icp_state[c] >= 0) is_dyn = icp_state[c] == 1 && tall;
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

    // Motion core of fused clusters: a static cluster whose dynamic voxels are diluted by static
    // structure is tested again on the voxels near its dynamic ones only.
    motion_core_.assign(params_.split_enable ? nvox : 0, 0);
    if (params_.split_enable) {
        const double r2 = params_.split_radius * params_.split_radius;
        for (int c = 0; c < n_clusters; ++c) {
            if (was_dynamic[c] || cluster_bboxes[c].is_dynamic_bbox() || dynamic_count[c] < params_.split_min_dyn) continue;
            std::vector<Eigen::Vector2d> dp;
            for (int j : members[c]) if (voxelmap.lookup_voxel(j).is_dynamic) dp.push_back(voxelmap.lookup_voxel(j).mean.head<2>());
            std::vector<int> core; int nd = 0; double vs = 0.0; double zmin = 1e9, zmax = -1e9;
            for (int j : members[c]) {
                const auto& v = voxelmap.lookup_voxel(j);
                bool near = v.is_dynamic;
                for (size_t q = 0; q < dp.size() && !near; ++q) near = (v.mean.head<2>() - dp[q]).squaredNorm() <= r2;
                if (!near) continue;
                core.push_back(j); if (v.is_dynamic) ++nd;
                if (!vis_frac_.empty()) vs += vis_frac_[j];
                zmin = std::min(zmin, v.mean.z()); zmax = std::max(zmax, v.mean.z());
            }
            if (core.empty()) continue;
            const double rc = static_cast<double>(nd) / core.size(), vc = vs / core.size();
            if (rc > params_.split_ratio && vc >= params_.split_min_vis && zmax - zmin + voxelmap.voxel_resolution() >= params_.min_obj_height)
                for (int j : core) motion_core_[j] = 1;
        }
    }

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
            if (!motion_core_.empty() && motion_core_[j] && !near_ground) { v.is_dynamic = true; return; }
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


double DynamicObjectRejectionCPU::icp_translation(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst,
                                                  Eigen::Vector3d& t, double& rms, double max_d)
{
    const double md2 = max_d * max_d;
    int in = 0; double se = 0.0;
    for (int it = 0; it < 10; ++it) {
        Eigen::Vector3d acc = Eigen::Vector3d::Zero(); in = 0; se = 0.0;
        for (const auto& p : src) {
            const Eigen::Vector3d q = p + t;
            double bd = md2; int bj = -1;
            for (size_t j = 0; j < dst.size(); ++j) { const double d = (dst[j] - q).squaredNorm(); if (d < bd) { bd = d; bj = static_cast<int>(j); } }
            if (bj < 0) continue;
            acc += dst[bj] - q; ++in; se += bd;
        }
        if (in < 3) { rms = 1e9; return 0.0; }
        const Eigen::Vector3d dt = acc / in;
        t += dt;
        if (dt.norm() < 1e-3) break;
    }
    rms = std::sqrt(se / std::max(in, 1));
    return static_cast<double>(in) / src.size();
}


// ===========================================================================
// Delayed decision (look-ahead free-space test)
// ===========================================================================

void DynamicObjectRejectionCPU::collect_delay_candidates(const gtsam_points::DynamicVoxelMapCPU& voxelmap,
                                                         const std::vector<BoundingBox>& cluster_bboxes, Pending& p) const
{
    const int nvox = nvox_of(voxelmap), nc = static_cast<int>(cluster_bboxes.size());
    const double r2 = params_.fut_max_range * params_.fut_max_range;
    std::vector<int> slot(nc, -1);
    for (int c = 0; c < nc; ++c) {
        const auto& b = cluster_bboxes[c];
        if (b.get_size().z() < params_.min_obj_height || b.get_center().head<2>().squaredNorm() > r2) continue;
        slot[c] = static_cast<int>(p.boxes.size()); p.boxes.emplace_back();
    }
    for (int j = 0; j < nvox && j < static_cast<int>(voxel_bboxes_.size()); ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (v.is_wall || v.is_ground || v.is_outlier) continue;
        const double g = hag_.empty() ? -1e9 : v.mean.z() - hag_[j];
        for (const auto& [c, in_base] : voxel_bboxes_[j]) {
            if (!in_base || c >= nc || slot[c] < 0) continue;
            for (size_t k = 0; k < v.voxel_points.size(); ++k) {
                if (k < v.voxel_times.size() && std::isnan(v.voxel_times[k])) continue;   // hybrid extra point
                if (v.voxel_points[k].z() - g < params_.point_ground_cut) continue;
                p.boxes[slot[c]].push_back(v.voxel_points[k].head<3>());
            }
        }
    }
}

void DynamicObjectRejectionCPU::finalize_delayed(DynamicRejectionResult& result)
{
    Pending p = std::move(pending_.front()); pending_.pop_front();
    const int D = params_.fut_delay, H = static_cast<int>(voxelmap_history_.size());
    const double res = params_.visibility_res_deg * M_PI / 180.0;
    const int cols = static_cast<int>(std::ceil(2.0 * M_PI / res)), rows = static_cast<int>(std::ceil(M_PI / res));
    while (range_img_history_.size() < voxelmap_history_.size())
        range_img_history_.push_back(build_range_image(*voxelmap_history_[range_img_history_.size()], rows, cols, res, std::max(0, params_.visibility_min_filter)));
    std::vector<int> idx; std::vector<Eigen::Isometry3d> T;
    for (int k = params_.fut_min_age; k <= D; k += std::max(1, params_.fut_age_step)) {
        const int h = H - 1 - (D - k);                  // history index of frame t+k (current frame t+D is last)
        if (h < 0 || h >= H) continue;
        idx.push_back(h); T.push_back(abs_pose_history_[h].inverse() * p.pose);
    }
    std::unordered_set<uint64_t> move;
    const auto key = [](const Eigen::Vector3d& q) {
        const float x = static_cast<float>(q.x()), y = static_cast<float>(q.y()), z = static_cast<float>(q.z());
        uint32_t a, b, c; std::memcpy(&a, &x, 4); std::memcpy(&b, &y, 4); std::memcpy(&c, &z, 4);
        return (static_cast<uint64_t>(a) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(b) << 21) ^ static_cast<uint64_t>(c) * 0xC2B2AE3D27D4EB4Full;
    };
    for (const auto& box : p.boxes) {
        if (box.size() < 3 || idx.empty()) continue;
        int freep = 0;
        for (const auto& q0 : box) {
            int votes = 0, seen = 0;
            for (size_t a = 0; a < idx.size(); ++a) {
                const Eigen::Vector3d q = T[a] * q0;
                const int c = std::min(cols - 1, static_cast<int>((std::atan2(q.y(), q.x()) + M_PI) / res));
                const int r = std::min(rows - 1, std::max(0, static_cast<int>((std::atan2(q.z(), q.head<2>().norm()) + M_PI / 2.0) / res)));
                const float fut = range_img_history_[idx[a]][r * cols + c];
                if (!std::isfinite(fut)) continue;
                ++seen;
                const double d = q.norm();
                if (fut > d + std::max(params_.visibility_abs_thr, params_.visibility_rel_thr * d)) ++votes;
            }
            if (seen > 0 && 2 * votes > seen) ++freep;
        }
        if (freep >= params_.fut_ratio * box.size())
            for (const auto& q0 : box) move.insert(key(q0));
    }
    result.has_delayed = true; result.delayed_stamp = p.stamp;
    if (move.empty() || !p.static_frame) { result.delayed_static_frame = p.static_frame; result.delayed_dynamic_frame = p.dynamic_frame; return; }
    const int kn = p.static_frame->k_neighbors;
    auto st = std::make_shared<PreprocessedFrame>(*p.static_frame);
    auto dy = p.dynamic_frame ? std::make_shared<PreprocessedFrame>(*p.dynamic_frame) : std::make_shared<PreprocessedFrame>();
    if (!p.dynamic_frame) { dy->stamp = p.static_frame->stamp; dy->scan_end_time = p.static_frame->scan_end_time; }
    st->points.clear(); st->intensities.clear(); st->times.clear(); st->neighbors.clear(); st->k_neighbors = 0;
    const auto& src = *p.static_frame;
    for (size_t i = 0; i < src.points.size(); ++i) {
        const bool m = move.count(key(src.points[i].head<3>())) > 0;
        auto& out = m ? *dy : *st;
        out.points.push_back(src.points[i]);
        if (i < src.intensities.size()) out.intensities.push_back(src.intensities[i]);
        if (i < src.times.size())       out.times.push_back(src.times[i]);
    }
    dy->k_neighbors = 0;
    st->k_neighbors = kn;
    if (kn > 0 && !st->points.empty()) st->neighbors = find_neighbors(st->points.data(), static_cast<int>(st->points.size()), kn);
    result.delayed_static_frame = st; result.delayed_dynamic_frame = dy;
}


// ===========================================================================
// point_evidence_labels()
// ===========================================================================
// Cluster-free labelling: points of a non-dynamic voxel with strong free-space evidence
// (past range images AND the free-space map, or the map only) become dynamic, so objects
// fused with static structure or without a cluster are still removed.

void DynamicObjectRejectionCPU::point_evidence_labels(const gtsam_points::DynamicVoxelMapCPU& voxelmap)
{
    const int nvox = nvox_of(voxelmap);
    if (static_cast<int>(fs_pt_.size()) != nvox) return;
    if (static_cast<int>(point_dyn_.size()) != nvox) point_dyn_.assign(nvox, {});
    const double r2max = params_.pe_max_range * params_.pe_max_range;
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        if (v.is_dynamic || v.is_wall || v.is_ground || v.is_outlier || fs_pt_[j].empty()) continue;
        if (v.mean.head<2>().squaredNorm() > r2max) continue;
        if (!hag_.empty() && hag_[j] < params_.ground_band_m) continue;
        const bool has_vis = static_cast<int>(vis_pt_.size()) == nvox && vis_pt_[j].size() == v.voxel_points.size();
        std::vector<uint8_t> lab(v.voxel_points.size(), 0);
        int n = 0;
        for (size_t k = 0; k < v.voxel_points.size(); ++k) {
            const bool strong = params_.pe_mode == 2 ? fs_pt_[j][k] != 0 : (fs_pt_[j][k] != 0 && has_vis && vis_pt_[j][k] == 1);
            if (strong) { lab[k] = 1; ++n; }
        }
        if (n >= params_.pe_min_points) point_dyn_[j] = std::move(lab);
    }
}


// ===========================================================================
// split_ground_points()
// ===========================================================================
// A dynamic voxel close to the ground (feet, wheels) usually also holds ground returns:
// its points below point_ground_cut above the local ground stay static.

void DynamicObjectRejectionCPU::split_ground_points(const gtsam_points::DynamicVoxelMapCPU& voxelmap)
{
    const int nvox = nvox_of(voxelmap);
    if (static_cast<int>(hag_.size()) != nvox) return;
    const double reach0 = params_.point_ground_cut + voxelmap.voxel_resolution();
    point_dyn_.assign(nvox, {});
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        const double cut = params_.point_ground_cut + params_.point_ground_cut_k * v.mean.head<2>().norm();
        if (!v.is_dynamic || !(hag_[j] < reach0 + cut - params_.point_ground_cut)) continue;
        if (v.mean.head<2>().norm() < params_.point_ground_cut_min_range) continue;   // close range: keep feet
        const double g = v.mean.z() - hag_[j];
        bool any_low = false;
        for (const auto& p : v.voxel_points) if (p.z() - g < cut) { any_low = true; break; }
        if (!any_low) continue;
        auto& lab = point_dyn_[j];
        lab.resize(v.voxel_points.size());
        for (size_t k = 0; k < v.voxel_points.size(); ++k) lab[k] = (v.voxel_points[k].z() - g >= cut) ? 1 : 0;
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