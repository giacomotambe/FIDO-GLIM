#pragma once

#include <deque>
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
    int    min_voxel_points;   ///< Minimum raw points per voxel to be scored. < 0 = use legacy points_limit formula.
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
    double w_distance;                  ///< Negative weight on voxel distance from origin. Score -= w_distance * dist. Suppresses far-range false positives.
    double w_velocity;                  ///< Weight on cluster EMA speed. Score += w_velocity * (speed - threshold). Positive above threshold, negative below.
    double velocity_static_threshold;   ///< Speed [m/s] below which the cluster is treated as static (score contribution becomes negative).
    double static_cluster_penalty_factor; ///< Score -= w_cluster * factor for voxels outside every dynamic bbox. (was hard-coded 0.5)
    double unmatched_dynamic_margin;      ///< Score assigned to unmatched voxels in a dynamic bbox = threshold + margin. (was hard-coded 1.0)
    // Long-baseline comparison (slow objects)
    int    long_baseline_frames;          ///< Also compare the voxel against frame t-k (k = this value). 0/1 = disabled.
    double w_shift_long;                  ///< Weight of the long-baseline centroid shift. Score uses max(w_shift*shift, w_shift_long*shift_long).
    // Final voxel assignment
    bool   keep_voxel_evidence;           ///< Keep voxels classified dynamic by per-voxel scoring even when outside every bbox.
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
    /// Per-voxel list of (bbox index, in_base) built once per frame.
    /// in_base == false means the voxel is only inside the velocity-inflated zone
    /// of a bbox that was dynamic at the time of indexing.
    std::vector<std::vector<std::pair<int, bool>>> voxel_bboxes_;
    

    std::vector<int> dynamic_voxels_neighbor_indices_;
    std::vector<int> dynamic_voxels;
    std::vector<int> neighbor_voxels_indices_;

    PreprocessedFrame::Ptr            last_dynamic_frame_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    std::unique_ptr<CloudCovarianceEstimation> covariance_estimation_;
    Eigen::Isometry3d last_pose_;

    std::vector<BoundingBox> last_cluster_bboxes_;

    double motion_scale_      = 1.0;  ///< Combined scale (translation + rotation), used for per-voxel threshold.
    double translation_scale_ = 0.0;  ///< params_.motion_threshold_scale    * robot_translation this frame.
    double rotation_scale_    = 0.0;  ///< params_.rotation_threshold_scale  * robot_rotation    this frame.
    VelocityInflationParams inflate_params_;  ///< Shared with BBOX mode, loaded from config_bbox_rejection.json.
};

}  // namespace glim