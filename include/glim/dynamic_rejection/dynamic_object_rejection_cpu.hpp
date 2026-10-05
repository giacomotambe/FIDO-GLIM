#pragma once

#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>
#include <glim/dynamic_rejection/voxel_filtering.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>
#include <glim/dynamic_rejection/cluster_extractor.hpp>


namespace glim {

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct DynamicObjectRejectionParamsCPU {
public:
    DynamicObjectRejectionParamsCPU();
    ~DynamicObjectRejectionParamsCPU();

public:
    // Scoring weights
    double dynamic_score_threshold;
    double tier1_threshold_factor;          ///< Multiplier on threshold for voxels inside a CONFIRMED dynamic cluster bbox (Tier 1). Must be > w_cluster/dynamic_score_threshold to avoid unconditional detection.
    double unconstrained_threshold_factor;  ///< Multiplier on threshold for voxels with no cluster and no dynamic history (Tier 3). Higher = fewer false positives.
    double memory_threshold_factor;         ///< Multiplier on threshold for voxels that were dynamic last frame but have no cluster bbox (Tier 2). < 1.0 for lower threshold.
    double w_shift;
    double w_mahalanobis;
    double w_neighbor;
    double w_cluster;
    double w_history;
    double w_history_dynamic;  ///< UNUSED (kept for config compatibility).
    double points_limit;       ///< Legacy min-points gate: voxel needs >= points_limit * voxel_res * 100 points (used when min_voxel_points < 0).
    int    min_voxel_points;
    int    sparse_voxel_mode;  ///< Voxels below the min-points gate: 0 = skip (legacy), 1 = score normally, 2 = visibility evidence only.   ///< Minimum raw points per voxel to be scored. < 0 = use legacy points_limit formula.
    // History
    double history_factor;     ///< UNUSED (kept for config compatibility).
    int    frame_num_memory;
    // Cluster propagation
    double cluster_propagation_threshold;
    // Motion-adaptive threshold
    double motion_threshold_scale;          ///< Threshold multiplier per meter of robot translation. Higher = less sensitive while moving.
    double rotation_threshold_scale;        ///< Threshold multiplier per radian of robot rotation. Higher = less sensitive while rotating.
    double min_shift_m;                 ///< Dead zone: centroid shifts below this value [m] contribute 0 to the score. Absorbs voxel-grid quantization artifacts.
    double cluster_motion_scale;          ///< Scales cluster_propagation_threshold by (1+translation_scale)^cluster_motion_scale.
    double cluster_rotation_scale;        ///< Scales cluster_propagation_threshold by (1+rotation_scale)^cluster_rotation_scale.
    double w_distance;                  ///< DEPRECATED (default 0, superseded by noise_deadzone_k). Negative weight on voxel distance from origin. Score -= w_distance * dist. Suppresses far-range false positives.
    double w_velocity;                  ///< Weight on cluster EMA speed. Score += w_velocity * (speed - threshold). Positive above threshold, negative below.
    double velocity_static_threshold;   ///< Speed [m/s] below which the cluster is treated as static (score contribution becomes negative).
    double static_cluster_penalty_factor; ///< Score -= w_cluster * factor for voxels outside every dynamic bbox. (was hard-coded 0.5)
    double unmatched_dynamic_margin;      ///< Score assigned to unmatched voxels in a dynamic bbox = threshold + margin. (was hard-coded 1.0)
    // Precision filters
    double ground_band_m;                 ///< Voxels lower than this above the local ground are never dynamic unless inside a confirmed-dynamic bbox tall >= min_obj_height.
    double ground_cell_m;                 ///< XY cell size for the local ground height estimate.
    double min_obj_height;                ///< A cluster can turn dynamic only if its bbox is at least this tall [m].
    double max_obj_bottom_hag;            ///< A cluster whose lowest voxel is higher than this above the local ground cannot turn dynamic [m] (0 = off).
    double max_obj_top_hag;               ///< ... nor one whose highest voxel is higher than this [m] (0 = off).
    double min_obj_speed;                 ///< ... and its estimated world-frame speed is at least this [m/s] (0 = off).
    double frame_max_dynamic_frac;        ///< If more than this fraction of a frame is dynamic, keep only confirmed-dynamic bboxes (anomalous frame). >= 1 = off.
    bool   inflated_requires_evidence;
    // Recall recovery
    double fast_confirm_ratio_factor;     ///< Remove a cluster in the same frame (no hysteresis) if ratio > factor * threshold ...
    double fast_confirm_min_vis;          ///< ... and its mean free-space fraction >= this. (factor <= 0 = off)
    double moving_vis_alt;                ///< A cluster passes the speed gate also if its mean free-space fraction >= this (> 1 = off).
    double foot_min_hag;
    // Point-level refinement (final step)
    double point_ground_cut;              ///< Points of dynamic voxels lower than this above the local ground stay static (<= 0 = off).
    bool   vac_enable;                    ///< Vacated-space evidence: the track moved and the place it occupied vac_gap_frames ago is now seen through.
    int    vac_gap_frames;                ///< Age of the reference scan (frames).
    double vac_min_disp;                  ///< Minimum XY displacement of the track centroid over the gap (m).
    double vac_frac;                      ///< Minimum fraction of the old points that are now seen through.
    int    vac_min_pts;                   ///< Minimum number of old points seen through.
    double vac_min_seen;                  ///< Minimum fraction of ALL the old points that are now seen through (occluded ones count against).
    int    vac_max_pts;                   ///< Points stored per track and frame.
    double vac_mid_tol;                   ///< > 0: constant-velocity check, max deviation of the mid-gap centroid from the midpoint (fraction of the displacement, + 0.1 m).
    bool   vac_require_tall;              ///< Apply min_obj_height to clusters confirmed by this test.
    double vac_max_len;                   ///< Only clusters up to this XY extent (m); 0 = no limit.
    bool   icp_enable;                    ///< Motion of vehicle-sized clusters from ICP against the same track icp_gap_frames ago.
    double icp_min_len;                   ///< Clusters longer than this [m] keep an ICP point history.
    int    icp_gap_frames;
    double icp_min_speed;                 ///< [m/s]
    double icp_max_rms;                   ///< [m]
    double icp_min_inlier;                ///< Inlier fraction after alignment.
    int    icp_max_pts;
    bool   fut_output_delayed;            ///< Async module outputs the delayed (refined) frames: mapping gets them fut_delay frames late.
    double min_obj_vis;                   ///< Min mean free-space fraction of a cluster to be dynamic in the frame (0 = off).
    double min_obj_vis_max_range;         ///< min_obj_vis applies only within this range [m].
    double view_change_k;                 ///< Dead-zone term for viewpoint change: k * voxel * min(1, baseline / range).
    int    rej_input_points;              ///< > 0: rejection runs on this many uniformly sampled raw points (labels transferred back).
    double label_transfer_radius;         ///< [m]
    int    fut_delay;                     ///< Delayed decision: frames of look-ahead (0 = off).
    int    fut_min_age;                   ///< Future scans used: t+fut_min_age, +fut_age_step, ..., <= t+fut_delay.
    int    fut_age_step;
    double fut_ratio;                     ///< Box becomes dynamic when this fraction of its points is seen free later.
    double fut_max_range;
    double large_box_len;                 ///< Clusters longer than this [m] need large_box_ratio / large_box_vis to be dynamic.
    double large_box_ratio;
    double large_box_vis;
    bool   vis_raw;                       ///< Past range images (free-space test) built from the raw scan instead of the voxel points.
    double vis_raw_max_range;             ///< Raw points beyond this range are ignored [m].
    int    vis_raw_max_points;            ///< Raw points used for the past range images (0 = all).
    bool   fs_raw;                        ///< Free-space map integrated from the raw scan.
    int    pe_mode;                       ///< Cluster-free point labels: 0 off, 1 range-image AND map free space, 2 map only.
    int    pe_min_points;                 ///< Min strong points in a voxel to relabel it.
    double pe_max_range;                  ///< Only within this range [m].
    bool   split_enable;                  ///< Motion core of static clusters (fused objects).
    double split_radius;                  ///< Voxels within this XY distance of a dynamic voxel form the core [m].
    int    split_min_dyn;                 ///< Min dynamic voxels in the cluster.
    double split_ratio;                   ///< Core dynamic ratio needed.
    double split_min_vis;                 ///< Core mean free-space fraction needed.
    bool   fs_enabled;                    ///< Use the world-frame free-space map as extra visibility evidence.
    double fs_res;                        ///< Cell size [m].
    int    fs_min_free;                   ///< A cell is "known free" after this many free frames ...
    double fs_free_ratio;                 ///< ... and free >= ratio * occ.
    int    fs_ray_stride;                 ///< Integrate one ray every N points.
    double fs_max_range;                  ///< Rays / queries only within this range [m].
    double fs_margin;                     ///< Ray stops max(margin, margin_rel * d) before the hit.
    double fs_margin_rel;
    int    fs_dilate;                     ///< Neighbourhood (cells) that must also be void.
    bool   fs_occ_static_only;            ///< Only points labelled static mark their cell occupied.
    double fs_decay;                      ///< Per-frame forgetting of free/occ counts (1 = none).
    bool   fs_flag_nb;                    ///< Neighbour-occupied flag stored per cell (1 lookup per query).
    bool   fs_dda;                        ///< Exact cell traversal (each crossed cell once).
    double point_ground_cut_min_range;    ///< The ground cut applies only beyond this range [m].
    double point_ground_cut_k;            ///< Range slope of the cut: cut(r) = point_ground_cut + k * r.
    bool   point_refine_enabled;          ///< Label individual points: grow from dynamic seeds, veto points with static free-space evidence.
    double point_grow_radius0;            ///< Growth radius at range 0 [m].
    double point_grow_radius_k;           ///< Growth radius increase per metre of range.
    double point_grow_radius_max;         ///< Growth radius cap [m].
    double point_bbox_margin;             ///< Growth stays inside confirmed-dynamic bboxes enlarged by this margin [m].
    bool   point_static_veto;             ///< Points whose past rays saw a surface at the same range stay static.
    int    point_knn;
    bool   point_grow_on_visibility;      ///< Also grow (outside bboxes) into points that appeared in free space.                     ///< Neighbours examined per grown point.                  ///< Near-ground / ground voxels inside the XY footprint of a confirmed tall dynamic bbox and higher than this above ground are dynamic (< 0 = off).    ///< Voxels in velocity-inflated / historical zones need their own evidence (score or visibility).
    // Hybrid density (extra far points from the raw scan, used only for the rejection)
    bool   hybrid_enabled;                ///< Augment the frame with raw points in [hybrid_min_range, hybrid_max_range].
    double hybrid_min_range;              ///< [m]
    double hybrid_max_range;              ///< [m]
    int    hybrid_max_points;             ///< Cap on added points per frame (<= 0 = no cap).
    // Pose handling
    bool   apply_lidar_imu_extrinsic;     ///< getPose() is T_world_imu: convert with T_imu_lidar from config_sensors (true in GLIM).
    Eigen::Isometry3d T_imu_lidar = Eigen::Isometry3d::Identity();
    double pose_err_trans_k;              ///< Expected centroid error per metre of ego translation over the baseline (added to sigma).
    double pose_err_rot_k;                ///< Expected angular pose error per radian of ego rotation over the baseline (x range, added to sigma).
    // Baseline comparison and noise-aware dead zone
    int    compare_baseline_frames;       ///< Compare each voxel against frame t-K (K = this value, 1 = previous frame).
    double noise_sigma0;                  ///< Expected centroid noise [m] at range 0.
    double noise_sigma_slope;             ///< Increase of expected centroid noise per metre of range.
    double noise_sigma_max;               ///< Cap of the expected centroid noise [m].
    double noise_deadzone_k;              ///< Dead zone = max(min_shift_m, k * sigma(range)).
    // Final voxel assignment
    bool   keep_voxel_evidence;           ///< Keep voxels classified dynamic by per-voxel scoring even when outside every bbox.
    // Range-image visibility check (per-point "appeared in previously free space")
    bool   visibility_enabled;            ///< Enable the range-image comparison against past scans.
    double w_visibility;                  ///< Score weight of the fraction of voxel points that appeared in free space.
    double visibility_res_deg;            ///< Angular resolution of the comparison range image [deg].
    double visibility_abs_thr;            ///< Point is "appeared" if r_now < r_past - max(abs_thr, rel_thr * r_now) [m].
    double visibility_rel_thr;            ///< Relative part of the threshold.
    int    visibility_min_age;            ///< Oldest/youngest past frames used: t-max_age .. t-min_age.
    int    visibility_max_age;
    int    visibility_age_step;      ///< Stride between the past scans used by the free-space test.
    int    history_frames;           ///< Frames used by history suppression (= configured frame_num_memory).
    double visibility_max_incidence_deg;
    int    visibility_min_filter;         ///< Half-size of the min filter on past range images (1 = 3x3, 0 = none).
    double visibility_keep_frac;          ///< Outside every bbox, keep a voxel dynamic if >= this fraction of its points appeared in free space (> 1 = off).
    int    visibility_keep_min_points;    ///< ... and it has at least this many points.  ///< Ignore current points seen at grazing incidence (surface normal vs ray) above this angle.
    // World-frame dynamic evidence grid (temporal consistency independent of track IDs)
    bool   evidence_enabled;              ///< Enable the world-frame evidence grid.
    double evidence_resolution;           ///< Cell size [m].
    double evidence_decay;                ///< Per-frame multiplicative decay of the evidence.
    double evidence_gain;                 ///< Evidence added when a voxel in the cell is classified dynamic.
    double evidence_on;                   ///< Cell switches ON when evidence >= this value.
    double evidence_off;                  ///< Cell switches OFF when evidence < this value (asymmetric hysteresis).
    // Permanent-static unlock
    double unlock_ratio_factor;           ///< A cluster shows "strong motion" when its dynamic ratio > factor * propagation threshold (also for locked clusters).
    // Misc
    int num_threads;
};

// ---------------------------------------------------------------------------
// Result of a single rejection pass
// ---------------------------------------------------------------------------

struct DynamicRejectionResult {
    /// Points classified as static, ready for odometry / mapping.
    PreprocessedFrame::Ptr static_frame;

    /// Points classified as dynamic (nullptr when none found).
    PreprocessedFrame::Ptr dynamic_frame;

    /// The classified voxelmap: wall voxels marked by WallFilter,
    /// dynamic voxels marked by the scorer. Useful for visualization.
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap;

    /// Delayed decision (fut_delay > 0): final labels of the frame received fut_delay calls ago,
    /// refined with the free-space test against the following scans. Empty until the delay is filled.
    bool   has_delayed = false;
    double delayed_stamp = 0.0;
    PreprocessedFrame::Ptr delayed_static_frame;
    PreprocessedFrame::Ptr delayed_dynamic_frame;
};

// ---------------------------------------------------------------------------
// DynamicObjectRejectionCPU
//
// Expected call sequence (caller owns both objects):
//
//   WallFilter             wall_filter(wall_cfg);
//   DynamicObjectRejectionCPU rejector(params);
//
//   // per frame:
//   WallFilterResult wf       = wall_filter.filter(*frame);
//   DynamicRejectionResult dr = rejector.reject(wf, frame);
//
// WallFilter::filter() already builds the DynamicVoxelMapCPU and marks wall
// voxels (is_wall = true).  reject() consumes that voxelmap directly — no
// second voxelization is performed.
// ---------------------------------------------------------------------------

class DynamicObjectRejectionCPU {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit DynamicObjectRejectionCPU(
        const DynamicObjectRejectionParamsCPU&   params               = DynamicObjectRejectionParamsCPU(),
        const std::shared_ptr<PoseKalmanFilter>&  pose_kalman_filter   = nullptr);

    // -----------------------------------------------------------------------
    // Primary API
    // -----------------------------------------------------------------------

    /**
     * @brief  Score voxels and split points into static / dynamic frames.
     *
     * Takes the WallFilterResult produced by WallFilter::filter() for the
     * current scan.  The voxelmap inside wf_result is used directly — no
     * additional voxelization is performed.
     *
     * Wall voxels (is_wall == true) are unconditionally kept as static.
     * All other voxels are scored against the previous frame's voxelmap and
     * classified as static or dynamic based on the configured weights and
     * threshold.
     *
     * On the first call (empty history) the voxelmap is stored as the
     * reference frame and source_frame is returned unchanged as static output.
     *
     * @param wf_result    Result of WallFilter::filter() for the current scan.
     * @param source_frame Original PreprocessedFrame; provides stamp,
     *                     scan_end_time, and k_neighbors for the output frames.
     * @return             DynamicRejectionResult{static_frame, dynamic_frame, voxelmap}.
     */
    DynamicRejectionResult reject(
        const WallFilterResult&         wf_result,
        const PreprocessedFrame::Ptr&   source_frame,
        std::vector<BoundingBox>&       cluster_bboxes,
        const std::vector<BoundingBox>& historical_bboxes = {});

    /**
     * @brief Hybrid density: return a copy of `frame` augmented with the raw points whose XY range is in
     *        [hybrid_min_range, hybrid_max_range] (at most hybrid_max_points, evenly strided). The added
     *        points carry time = NaN and are dropped from the static/dynamic output frames by reject(),
     *        so odometry receives exactly the original points. Returns `frame` itself when disabled or
     *        when frame->raw_points is not available.
     */
    static PreprocessedFrame::Ptr make_hybrid_frame(const PreprocessedFrame::Ptr& frame, const DynamicObjectRejectionParamsCPU& params);
    /// Input of the rejection when rej_input_points > 0: a uniform random subsample of the raw scan
    /// (the odometry downsampling - random grid - leaves too few points on near objects).
    /// Falls back to make_hybrid_frame(frame) when disabled or no raw scan is attached.
    static PreprocessedFrame::Ptr make_rejection_frame(const PreprocessedFrame::Ptr& frame, const DynamicObjectRejectionParamsCPU& params);
    /// Copies the labels of the rejection input (static/dynamic) onto the points of `target`
    /// (nearest labelled point within label_transfer_radius; farther points stay static).
    static void transfer_labels(const PreprocessedFrame::Ptr& target, DynamicRejectionResult& result, const DynamicObjectRejectionParamsCPU& params);
    const DynamicObjectRejectionParamsCPU& params() const { return params_; }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    PreprocessedFrame::Ptr get_last_dynamic_frame() const {
        return last_dynamic_frame_;
    }

    gtsam_points::DynamicVoxelMapCPU::Ptr get_last_voxelmap() const {
        return voxelmap_history_.empty() ? nullptr : voxelmap_history_.back();
    }


private:
    // -----------------------------------------------------------------------
    // Pipeline steps (called in order inside reject())
    // -----------------------------------------------------------------------

    /// Update the world-frame evidence grid with this frame's labels and relabel voxels from the cell state.
    void apply_world_evidence(gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor);

    /// Fill vis_frac_ (per voxel fraction of points that appeared in previously free space).
    void compute_visibility(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor);

    /// Build voxel_bboxes_ (single O(nvox * n_bbox) pass, parallel).
    void index_voxel_bboxes(
        const gtsam_points::DynamicVoxelMapCPU& voxelmap,
        const std::vector<BoundingBox>&         cluster_bboxes);

    /// Look up the voxel of `current` whose centroid is closest to p (in the frame of `map`),
    /// exact cell first then 26-neighbour fallback. Returns -1 if none within 2 voxels.
    int match_voxel(const gtsam_points::DynamicVoxelMapCPU& map, const Eigen::Vector3d& p, double voxel_res) const;

    /// Per-voxel scoring against the previous voxelmap.
    /// Populates dynamic_voxels_indices_ and dynamic_voxels_neighbor_indices_.
    void score_voxels(
        gtsam_points::DynamicVoxelMapCPU&       current,
        const gtsam_points::DynamicVoxelMapCPU& previous,
        const std::vector<BoundingBox>&         cluster_bboxes,
        const Eigen::Isometry3d&                T_delta_pose);

    /// Boost the dynamic score of direct neighbours of confirmed dynamic voxels
    /// and re-apply the threshold.
    void propagate_to_neighbors(
        gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox);
    
    /// If a percentage of voxels in a cluster are dynamic, mark the whole cluster as dynamic.
    /// historical_bboxes: past-frame bboxes of confirmed-dynamic tracks (current sensor frame),
    /// used for an additional contains / contains_inflated check for voxels outside all cluster bboxes.
    void propagate_to_clusters(
        gtsam_points::DynamicVoxelMapCPU& voxelmap,
        std::vector<BoundingBox>&         cluster_bboxes,
        const std::vector<BoundingBox>&   historical_bboxes = {});

    /// Iterate all voxels and append their raw points to the appropriate bucket.
    void collect_points(
        const gtsam_points::DynamicVoxelMapCPU& voxelmap,
        std::vector<Eigen::Vector4d>& static_pts,
        std::vector<double>&          static_int,
        std::vector<double>&          static_tim,
        std::vector<Eigen::Vector4d>& dynamic_pts,
        std::vector<double>&          dynamic_int,
        std::vector<double>&          dynamic_tim) const;

    /// Assemble a PreprocessedFrame from point/intensity/time vectors,
    /// recomputing neighbor indices when source.k_neighbors > 0.
    PreprocessedFrame::Ptr build_frame(
        std::vector<Eigen::Vector4d> points,
        std::vector<double>          intensities,
        std::vector<double>          times,
        const PreprocessedFrame&     source) const;

    /// KNN neighbor search used by build_frame().
    std::vector<int> find_neighbors(
        const Eigen::Vector4d* points, int num_points, int k) const;

    /// Return indices of the 26 direct voxel neighbours of a given centroid.
    std::vector<int> get_neighbor_voxels(
        const gtsam_points::DynamicVoxelMapCPU& voxelmap,
        const Eigen::Vector4d& mean) const;


private:
    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    DynamicObjectRejectionParamsCPU params_;

    /// Ring buffer of past voxelmaps (oldest → newest).
    std::deque<gtsam_points::DynamicVoxelMapCPU::Ptr> voxelmap_history_;
    /// pose_history_[i] maps points from the sensor frame of voxelmap_history_[i]
    /// into the sensor frame of voxelmap_history_[i-1] (Identity for the first frame).
    std::deque<Eigen::Isometry3d> pose_history_;
    std::deque<Eigen::Isometry3d> abs_pose_history_;   ///< T_world_sensor of each voxelmap_history_ entry.
    std::vector<float> vis_frac_;
    /// Local free-space map in the world frame (DUFOMap-like): per cell, number of frames in which a
    /// ray crossed it (free) or ended in it (occ). Long memory without keeping the past scans.
    struct FsCell { float free = 0.f, occ = 0.f; uint32_t last_free = 0, last_occ = 0; uint16_t near = 0; bool marked = false; };
    std::vector<float> fs_pow_;
    float fs_decayed(float v, uint32_t last) const;
    std::unordered_map<int64_t, FsCell> fs_map_;
    std::vector<uint64_t> fs_seen_;   ///< Per-frame bitmap of cells already marked free (sensor-centred).
    uint32_t fs_frame_ = 0;
    void fs_query(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor);
    void fs_integrate(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const Eigen::Isometry3d& T_world_sensor);
    void fs_integrate_points(const std::vector<Eigen::Vector4d>& pts, const Eigen::Isometry3d& T_world_sensor);
    void push_raw_range_image(const PreprocessedFrame::Ptr& frame);
    std::vector<float> make_raw_range_image(const PreprocessedFrame::Ptr& frame) const;
    std::vector<float> hag_;
    std::vector<std::vector<uint8_t>> fs_pt_;     ///< Per voxel, per point: 1 if the free-space map says the point fills known free space.
    std::vector<char> motion_core_;
    struct Pending {
        PreprocessedFrame::Ptr static_frame, dynamic_frame;
        double stamp; Eigen::Isometry3d pose;
        std::vector<std::vector<Eigen::Vector3d>> boxes;   ///< Candidate objects: non-ground points, sensor frame.
    };
    std::deque<Pending> pending_;
    void collect_delay_candidates(const gtsam_points::DynamicVoxelMapCPU& voxelmap, const std::vector<BoundingBox>& cluster_bboxes, Pending& p) const;
    void finalize_delayed(DynamicRejectionResult& result);
    struct IcpEntry { int frame; double stamp; std::vector<Eigen::Vector3d> pts; };
    std::unordered_map<int, std::deque<IcpEntry>> vac_hist_;   ///< Per track: recent world-frame point sets (vacated-space test).
    std::vector<float> cur_range_img_;                         ///< Range image of the current scan (built before the cluster decision when vac_enable).
    std::unordered_map<int, std::deque<IcpEntry>> icp_hist_;   ///< Per track: recent world-frame point sets.
    int icp_frame_ = 0;
    double cur_stamp_ = 0.0;
    Eigen::Isometry3d cur_pose_ = Eigen::Isometry3d::Identity();
    /// Translation-only ICP of src onto dst (world frame). Returns inlier fraction, fills t and rms.
    static double icp_translation(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst,
                                  Eigen::Vector3d& t, double& rms, double max_d);               ///< Per voxel: part of the motion core of a (fused) static cluster.
    void point_evidence_labels(const gtsam_points::DynamicVoxelMapCPU& voxelmap);
    std::vector<std::vector<uint8_t>> vis_pt_;     ///< Per voxel, per point: 0 unknown, 1 appeared in free space, 2 consistent with the past.
    std::vector<std::vector<uint8_t>> point_dyn_;  ///< Per voxel, per point final label (empty = use voxel label).
    std::vector<BoundingBox> confirmed_bboxes_;    ///< Confirmed-dynamic bboxes of this frame.
    void split_ground_points(const gtsam_points::DynamicVoxelMapCPU& voxelmap);
    void refine_points(const gtsam_points::DynamicVoxelMapCPU& voxelmap);              ///< Per-voxel height above local ground [m].
    void compute_ground_heights(const gtsam_points::DynamicVoxelMapCPU& voxelmap);
    std::deque<std::vector<float>> range_img_history_;  ///< Range image of each voxelmap_history_ entry (own frame, 3x3 min).
    /// Per-voxel list of (bbox index, in_base) built once per frame.
    /// in_base == false means the voxel is only inside the velocity-inflated zone
    /// of a bbox that was dynamic at the time of indexing.
    std::vector<std::vector<std::pair<int, bool>>> voxel_bboxes_;
    struct EvidenceCell { float e = 0.f; int last = 0; bool on = false; };
    std::unordered_map<int64_t, EvidenceCell> evidence_;
    int frame_idx_ = 0;
    

    std::vector<int> dynamic_voxels_neighbor_indices_;
    std::vector<int> dynamic_voxels;
    std::vector<int> neighbor_voxels_indices_;

    PreprocessedFrame::Ptr            last_dynamic_frame_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation_;
    Eigen::Isometry3d last_pose_;
    Eigen::Isometry3d lidar_pose() const { return pose_kalman_filter_->getPose() * params_.T_imu_lidar; }

    std::vector<BoundingBox> last_cluster_bboxes_;

    double motion_scale_      = 1.0;  ///< Combined scale (translation + rotation), used for per-voxel threshold.
    double translation_scale_ = 0.0;  ///< params_.motion_threshold_scale    * robot_translation this frame.
    double rotation_scale_    = 0.0;  ///< params_.rotation_threshold_scale  * robot_rotation    this frame.
    VelocityInflationParams inflate_params_;  ///< Shared with BBOX mode, loaded from config_bbox_rejection.json.
};

}  // namespace glim