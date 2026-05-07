#pragma once

#include <deque>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>

namespace glim {

// ===========================================================================
// Track
// ===========================================================================

enum class PermanentState { NONE, DYNAMIC, STATIC };

struct Track {
    int             id;
    Eigen::Vector3d center;       ///< Last known center (current sensor frame)
    BoundingBox     last_bbox;    ///< Last matched bbox (for overlap gating)
    int             age;
    int             missed_frames;
    int             dynamic_frames = 0;   ///< Consecutive frames confirmed dynamic by propagate_to_clusters()
    int             static_frames  = 0;   ///< Consecutive frames confirmed static by propagate_to_clusters()
    PermanentState  permanent_state = PermanentState::NONE;
};

// ===========================================================================
// DynamicClusterExtractorParams
// ===========================================================================
struct DynamicClusterExtractorParams {
public:
    DynamicClusterExtractorParams();
    ~DynamicClusterExtractorParams();

public:
    // --- DBSCAN ---
    double eps_voxel_factor;   ///< eps = eps_voxel_factor * voxel_resolution. Default: 2.0
    int    min_pts;            ///< Min neighbours to be a core point. Default: 1
    int    knn_max_neighbors;  ///< Max neighbours in knn range query. Default: 64
    int    min_cluster_voxels; ///< Min voxels to keep a cluster. Default: 2
    int    min_points_for_bbox;///< Min raw points to generate a bbox. Default: 20

    // --- Bbox filters ---
    double bbox_min_extent;    ///< Min side length [m]. Default: 0.0
    double bbox_max_extent;    ///< Max side length [m]. Default: 1e9
    double bbox_min_volume;    ///< Min volume [m³]. Default: 0.0
    double bbox_max_volume;    ///< Max volume [m³]. Default: 1e9

    // --- NMS ---
    double cluster_iou_threshold; ///< IoU threshold for NMS. Default: 0.5

    // --- Pre-tracking peer merge ---
    /// Max center-to-center distance [m] to merge two current-frame bboxes.
    /// Set to 0.0 to disable. Default: 0.0
    double peer_merge_distance;

    // --- Tracking ---
    double track_match_distance; ///< Max center distance for track association [m]. Default: 1.5
    double track_match_iou;      ///< Min overlap for track association. Default: 0.3
    int    track_max_missed;     ///< Frames without match before track deletion. Default: 3
    int    min_dynamic_frames;        ///< Consecutive confirmed-dynamic frames before bbox is flagged as dynamic. Default: 3
    int    permanent_dynamic_frames;  ///< Consecutive dynamic frames to lock track as permanently dynamic. 0 = disabled. Default: 10
    int    permanent_static_frames;   ///< Consecutive static frames to lock track as permanently static.  0 = disabled. Default: 10
};

// ===========================================================================
// DynamicClusterExtractor
// ===========================================================================
class DynamicClusterExtractor {
public:
    using VoxelMapPtr = gtsam_points::DynamicVoxelMapCPU::Ptr;
    using ClusterMap  = std::vector<int>;

    DynamicClusterExtractor(const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter = nullptr);
    explicit DynamicClusterExtractor(const DynamicClusterExtractorParams& params);
    ~DynamicClusterExtractor() = default;

    ClusterMap cluster_voxels(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

    std::vector<std::vector<Eigen::Vector4d>> build_point_clusters(
        VoxelMapPtr       voxelmap,
        const ClusterMap& cluster_map,
        int               num_clusters) const;

    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters) const;

    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters,
        int min_points) const;

    const DynamicClusterExtractorParams& params() const { return params_; }

    std::vector<BoundingBox> extract_clusters(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap);

private:
    bool createAABB(const std::vector<Eigen::Vector4d>& cluster, BoundingBox& out_bbox) const;

    /// Merge nearby current-frame bboxes (iterative, by center distance).
    std::vector<BoundingBox> merge_nearby_clusters(const std::vector<BoundingBox>& bboxes) const;

    /// Match bboxes to tracks by overlap, assign IDs, create/prune tracks.
    void update_tracks(std::vector<BoundingBox>& bboxes, const Eigen::Isometry3d& T_to_current);

public:
    /// Called after reject() to feed propagate_to_clusters() results back into track dynamic counters.
    /// Increments dynamic_frames for tracks whose bbox was confirmed dynamic; resets to 0 otherwise.
    void update_dynamic_feedback(const std::vector<BoundingBox>& post_rejection_bboxes);

private:
    DynamicClusterExtractorParams     params_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    Eigen::Isometry3d                 last_pose_;

    std::vector<Track> tracks_;
    int                next_track_id_ = 0;
};

} // namespace glim
