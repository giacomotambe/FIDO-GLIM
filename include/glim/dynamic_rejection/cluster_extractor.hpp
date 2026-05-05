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
// Track — lightweight bounding-box track for temporal consistency
// ===========================================================================
struct Track {
    int             id;             ///< Unique track ID
    Eigen::Vector3d center;         ///< Last known center (current sensor frame)
    Eigen::Vector3d velocity;       ///< Estimated velocity (delta per frame)
    BoundingBox     last_bbox;      ///< Last matched bbox (for IoU gating)
    int             age;            ///< Frames the track has been alive
    int             missed_frames;  ///< Consecutive frames without a match
    int             slow_frames;     ///< Consecutive frames with speed < velocity_static_threshold
    double          dynamic_score;   ///< EMA score in [0,1]: 0=static, 1=dynamic
    bool            is_dynamic;      ///< Final stable classification (score >= threshold)
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

    /// Moltiplicatore applicato alla voxel_resolution per ottenere eps.
    /// eps = eps_voxel_factor * voxel_resolution
    /// Default: 2.0
    double eps_voxel_factor;

    /// Numero minimo di vicini (escluso se stesso) per essere core-point.
    /// Default: 1
    int min_pts;

    /// Numero massimo di vicini cercati nella knn_search della range query.
    /// Default: 64
    int knn_max_neighbors;

    // --- Filtro dimensione cluster ---

    /// Numero minimo di voxel per tenere un cluster.
    /// Default: 2
    int min_cluster_voxels;

    /// Numero minimo di punti raw per generare una bounding box.
    /// Default: 20
    int min_points_for_bbox;

    // --- Filtro dimensionale sulle bounding box ---

    /// Dimensione minima ammessa lungo ciascun asse dell'OBB [m].
    /// Un'OBB viene scartata se il suo lato più corto è < bbox_min_extent.
    /// Utile per eliminare cluster rumorosi troppo piccoli.
    /// Default: 0.0  (nessun filtro)
    double bbox_min_extent;

    /// Dimensione massima ammessa lungo ciascun asse dell'OBB [m].
    /// Un'OBB viene scartata se il suo lato più lungo è > bbox_max_extent.
    /// Utile per eliminare falsi positivi enormi (es. muri non classificati).
    /// Default: 1e9  (nessun filtro)
    double bbox_max_extent;

    /// Volume minimo ammesso [m³].
    /// Default: 0.0  (nessun filtro)
    double bbox_min_volume;

    /// Volume massimo ammesso [m³].
    /// Default: 1e9  (nessun filtro)
    double bbox_max_volume;
    // Cluster classification
    double cluster_distance_threshold;
    double cluster_iou_threshold;

    /// Numero di frame passati da conservare nella storia per la classificazione.
    /// Default: 5
    int history_size;

    /// Numero minimo di frame storici in cui un cluster deve matchare per essere
    /// classificato come statico (deve includere il frame immediatamente precedente).
    /// Default: 3
    int min_static_history_matches;


    double containment_margin;  // default 0.1m
    double merge_volume_ratio;  // default 1.5

    /// Max center-to-center distance [m] for merging two current-frame bboxes
    /// into a single larger one.  Set to 0.0 to disable peer merging.
    /// Default: 0.0
    double peer_merge_distance;

    // --- Cluster tracking ---

    /// Max center distance (predicted) to associate a bbox to a track [m].
    /// Default: 1.5
    double track_match_distance;
    double track_match_iou;  // default 0.3

    /// A track is deleted after this many consecutive unmatched frames.
    /// Default: 3
    int track_max_missed;

    /// Consecutive frames raw-classified dynamic required to flip stable state
    /// from static → dynamic.  Default: 2
    int hysteresis_dynamic_n;

    /// Consecutive frames raw-classified static required to flip stable state
    /// from dynamic → static.  Default: 3
    int hysteresis_static_m;

    // --- Velocity-based static override ---

    /// Speed below which a track is considered slow [m/frame].
    /// Default: 0.05
    double velocity_static_threshold;

    /// Consecutive slow frames required to force the track to static.
    /// Default: 3
    int velocity_static_frames;

    // --- EMA-based classification ---

    /// EMA smoothing factor for dynamic_score updates. Larger = faster response.
    /// Default: 0.3
    double ema_alpha;

    /// Score in [0,1] above which a track is classified DYNAMIC. Default: 0.6
    double dynamic_score_threshold;

    /// Speed above which a track is definitely DYNAMIC [m/frame]. Default: 0.2
    double velocity_dynamic_threshold;

    /// Tracks younger than this many frames are kept DYNAMIC (uncertain state).
    /// Default: 3
    int min_track_age;

    /// Fraction of history frames that must match for a track to be called static.
    /// Replaces the fixed min_static_history_matches count. Default: 0.6
    double static_history_ratio;

    /// Low-pass filter coefficient for velocity smoothing (0=instant, 1=frozen). Default: 0.8
    double velocity_beta;

    /// Additional threshold scaling per metre of ego-motion (m/frame → scale factor).
    /// scale = 1 + motion_scale_factor * ||translation||. Default: 2.0
    double motion_scale_factor;

    /// Distance below which a track–bbox pair is always accepted, regardless of IoU. Default: 0.5
    double track_match_distance_strict;
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

    /**
     * @brief  DBSCAN sui centroidi dei voxel NON-wall.
     */
    ClusterMap cluster_voxels(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

    // -----------------------------------------------------------------------
    // Utilities post-rejection
    // -----------------------------------------------------------------------

    std::vector<std::vector<Eigen::Vector4d>> build_point_clusters(
        VoxelMapPtr       voxelmap,
        const ClusterMap& cluster_map,
        int               num_clusters) const;

    /// Genera OBB per ogni cluster con abbastanza punti, applicando i filtri
    /// dimensionali configurati (bbox_min/max_extent, bbox_min/max_volume).
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters) const;

    /// Overload con soglia minima punti esplicita (usa comunque i filtri dimensionali).
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters,
        int min_points) const;

    const DynamicClusterExtractorParams& params() const { return params_; }

    std::vector<BoundingBox> extract_clusters(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap);

    /// Returns a snapshot of the history as a vector-of-vectors (one per frame age).
    /// Index 0 = most recent frame, index N-1 = oldest.
    std::deque<std::vector<BoundingBox>> get_cluster_history_snapshot() const {
        std::deque<std::vector<BoundingBox>> out;
        for (const auto& [pose, bboxes] : cluster_history_)
            out.push_back(bboxes);
        return out;
    }

private:
    /// Crea l'OBB per un singolo cluster.  Ritorna true se l'OBB supera i
    /// filtri dimensionali configurati, false se deve essere scartata.
    bool createOBB(const std::vector<Eigen::Vector4d>& cluster,
                   BoundingBox& out_bbox) const;
    
    void classify_clusters(
        Eigen::Isometry3d T_delta_pose,
        std::vector<BoundingBox>& cluster_bboxes);

    /// Match current bboxes to existing tracks (nearest-neighbour on predicted
    /// center), update track kinematics, create new tracks for unmatched bboxes,
    /// and prune tracks that have been missing too long.
    /// Attaches the matched track ID to each bbox via BoundingBox::set_track_id().
    /// @param bboxes       Current-frame bboxes (in current sensor frame).
    /// @param T_to_current Transform that brings previous-frame coords into the
    ///                     current sensor frame (= T_delta_pose.inverse()).
    void update_tracks(std::vector<BoundingBox>& bboxes,
                       const Eigen::Isometry3d&  T_to_current,
                       double                    dist_scale);

    std::vector<BoundingBox> merge_with_history(
        const std::vector<BoundingBox>& current_bboxes,
        const std::deque<std::vector<BoundingBox>>& history,
        const std::unordered_set<int>& protected_track_ids) const;
    

private:
    DynamicClusterExtractorParams params_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    Eigen::Isometry3d last_pose_;

    /// Historia: (T_world_sensor at capture time, bboxes in sensor frame at that time).
    /// Bboxes are stored as-is; transforms are computed on demand to avoid drift.
    std::deque<std::pair<Eigen::Isometry3d, std::vector<BoundingBox>>> cluster_history_;

    /// Active tracks, maintained across frames.
    std::vector<Track> tracks_;
    /// Monotonically increasing track ID counter.
    int next_track_id_ = 0;


};

} // namespace glim