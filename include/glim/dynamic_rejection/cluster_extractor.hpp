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
    int             unlock_frames  = 0;   ///< Consecutive strong-motion frames while PERMANENT_STATIC
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  ///< EMA-smoothed velocity [m/s] in current sensor frame
    std::deque<BoundingBox> bbox_history;  ///< Recent past bboxes in current sensor frame (oldest front, newest back)
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
    std::string env_type; ///< "INDOOR" or "OUTDOOR", used to select peer_merge_distance from config.

    // --- Tracking ---
    double track_match_distance; ///< Max center distance for track association [m]. Default: 1.5
    double track_match_iou;      ///< Min overlap for track association. Default: 0.3
    int    track_max_missed;     ///< Frames without match before track deletion. Default: 3
    int    min_dynamic_frames;        ///< Consecutive confirmed-dynamic frames before bbox is flagged as dynamic. Default: 3
    int    permanent_dynamic_frames;  ///< Consecutive dynamic frames to lock track as permanently dynamic. 0 = disabled. Default: 10
    int    permanent_static_frames;   ///< Consecutive static frames to lock track as permanently static.  0 = disabled. Default: 10
    int    track_bbox_history_size;   ///< Number of past bboxes stored per track for historical inflated-zone checks. Default: 5
    bool   use_motion_prediction;     ///< Predict track position with constant velocity before association. Default: true
    double fast_track_speed;          ///< XY speed [m/s] above which a track uses fast_track_min_dynamic_frames. <= 0 = disabled. Default: 0 (1-frame hysteresis caused large false positives on indoor data)
    int    fast_track_min_dynamic_frames; ///< Hysteresis for fast tracks (lower latency). Default: 1
    bool   use_centroid_association;  ///< Associate by point-centroid distance (Hungarian) instead of bbox overlap (greedy). Default: true
    double assoc_gate_base;           ///< Association gate [m] at range 0 for a track seen in the previous frame. Default: 0.8
    double assoc_gate_per_range;      ///< Gate increase per metre of range. Default: 0.03
    double assoc_gate_per_missed;     ///< Gate increase per missed frame. Default: 0.3
    Eigen::Isometry3d T_imu_lidar = Eigen::Isometry3d::Identity();  ///< from config_sensors when apply_lidar_imu_extrinsic
    int    release_static_frames;
    // Performance / region of interest
    bool   dbscan_voxel_hash;         ///< Neighbour search through a voxel-coordinate hash instead of a KD-tree.
    int    dbscan_cell_range;         ///< Hash neighbourhood half-size in cells (1 = 26-neighbourhood, 2 = exact for eps <= 1.5 voxels).
    double cluster_max_range;         ///< Voxels farther than this (XY) are not clustered (<= 0 = off).
    double cluster_max_height;        ///< Voxels higher than this above the local ground are not clustered (<= 0 = off).
    double cluster_ground_cell;
    // Fragmentation
    double eps_range_k;               ///< eps(r) = max(eps, eps_range_k * r): sparse far objects stay connected (0 = off).
    double frag_merge_gap;            ///< Merge bboxes whose AABB gap <= frag_merge_gap + frag_merge_gap_k * r (< 0 = off).
    double frag_merge_gap_k;
    double frag_max_len, frag_max_wid, frag_max_hgt;  ///< A merged bbox must stay within these sorted dimensions [m].       ///< XY cell size of the local ground estimate used by cluster_max_height.     ///< Consecutive static frames needed to switch a confirmed-dynamic track back to static (asymmetric hysteresis). 0 = immediate (legacy). Default: 0 (5 raised false removals on indoor data)
    int    permanent_unlock_frames;   ///< Consecutive strong-motion frames that release a PERMANENT_STATIC track. 0 = never. Default: 2
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

    /// stamp: scan timestamp in seconds, used to compute dt for velocity estimation.
    std::vector<BoundingBox> extract_clusters(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap, double stamp = 0.0);

private:
    bool createAABB(const std::vector<Eigen::Vector4d>& cluster, BoundingBox& out_bbox) const;

    /// Merge nearby current-frame bboxes (iterative, by center distance).
    std::vector<BoundingBox> merge_nearby_clusters(const std::vector<BoundingBox>& bboxes) const;

    /// Read-only pass: propagate track ID and is_dynamic from track history onto fresh bboxes
    /// BEFORE NMS/merge so that bboxes with distinct known tracks are protected from
    /// suppression or absorption by each other.
    void label_bboxes_from_tracks(std::vector<BoundingBox>& bboxes, const Eigen::Isometry3d& T_to_current, double dt) const;
    /// Consecutive confirmed-dynamic frames required before the track is flagged dynamic.
    int  required_dynamic_frames(const Track& t) const;
    /// Track bbox moved to the current frame and shifted by velocity * dt (constant-velocity prediction).
    BoundingBox predicted_bbox(const Track& t, const Eigen::Isometry3d& T_to_current, double dt) const;

    /// Match bboxes to tracks by overlap, assign IDs, create/prune tracks.
    /// dt: time elapsed since last call [s], used for velocity estimation (0 = skip).
    void update_tracks(std::vector<BoundingBox>& bboxes, const Eigen::Isometry3d& T_to_current, double dt);

public:
    /// Called after reject() to feed propagate_to_clusters() results back into track dynamic counters.
    /// Increments dynamic_frames for tracks whose bbox was confirmed dynamic; resets to 0 otherwise.
    void update_dynamic_feedback(const std::vector<BoundingBox>& post_rejection_bboxes);

    /// Returns all historical bboxes (already in current sensor frame) for tracks that are
    /// currently confirmed dynamic (dynamic_frames >= min_dynamic_frames or permanently dynamic).
    /// Used by propagate_to_clusters() for the historical inflated-zone check.
    std::vector<BoundingBox> get_dynamic_track_history() const;

private:
    DynamicClusterExtractorParams     params_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    Eigen::Isometry3d                 last_pose_;

    std::vector<Track> tracks_;
    int                next_track_id_ = 0;
    double             last_stamp_    = 0.0;
};

} // namespace glim
