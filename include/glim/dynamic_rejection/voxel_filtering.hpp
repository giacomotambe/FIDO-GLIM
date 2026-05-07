#pragma once

#include <memory>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>  // adjust path
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/wall_bbox.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>

namespace glim {

struct PreprocessedFrame;

// ---------------------------------------------------------------------------
// Piano 3D: normal.dot(p) + d = 0,  ||normal|| = 1
// ---------------------------------------------------------------------------
struct PlaneModel {
    Eigen::Vector3d normal;
    double d;

    double distance(const Eigen::Vector3d& pt) const {
        return std::abs(normal.dot(pt) + d);
    }
};

// ---------------------------------------------------------------------------
// Risultato di WallFilter::filter()
// ---------------------------------------------------------------------------
struct WallFilterResult {
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap;
    std::vector<PlaneModel> wall_planes;
    std::vector<PlaneModel> floor_planes;
    int num_wall_voxels  = 0;
    int num_total_voxels = 0;
};


// ---------------------------------------------------------------------------
// Configurazione
// ---------------------------------------------------------------------------

struct WallFilterConfig {
    double voxel_resolution;
    int    ransac_max_iterations;
    double ransac_inlier_threshold;
    int    ransac_min_inliers;
    double ransac_confidence;
    double wall_vertical_angle_deg;
    double floor_ceiling_angle_deg;
    int    floor_min_inliers;
    int    max_planes;
    double wall_bbox_max_aspect_ratio;
    double floor_height_threshold;

    // --- Polar floor segmentation ---
    bool   floor_polar_enabled;
    double floor_polar_r_bin;
    double floor_polar_theta_deg;
    double floor_polar_slope_threshold;
    double floor_polar_seed_min_r;
    double floor_polar_seed_max_r;
    double floor_polar_height_margin;
    int    floor_polar_k_lowest;
    double floor_polar_flatness_threshold;
    double floor_polar_max_ground_height;
    double floor_polar_max_vertical_spread;

    Eigen::Isometry3d T_lidar_imu;
    Eigen::Isometry3d T_imu_lidar;

    WallFilterConfig();   // reads from config file
    ~WallFilterConfig();
};

// ---------------------------------------------------------------------------
// WallFilter
//
// Pipeline:
//   1. Voxelizza PreprocessedFrame → DynamicVoxelMapCPU (unica voxelizzazione)
//   2. Estrai centroidi voxel
//   3. RANSAC iterativo sui centroidi → piani dominanti
//   4. Classifica ogni piano: parete se normale quasi orizzontale
//   5. Marca i voxel-parete nella voxelmap (is_dynamic = true come sentinel,
//      oppure con flag is_wall se aggiunto a DynamicGaussianVoxel)
//   6. Restituisce WallFilterResult con la voxelmap già pronta per
//      DynamicObjectRejectionCPU — nessuna doppia voxelizzazione.
//
// DynamicObjectRejectionCPU deve essere adattato per accettare una
// voxelmap pre-costruita invece di un PreprocessedFrame raw.
// ---------------------------------------------------------------------------
class WallFilter {
public:
    using Ptr      = std::shared_ptr<WallFilter>;
    using ConstPtr = std::shared_ptr<const WallFilter>;

    explicit WallFilter(const WallFilterConfig& config = WallFilterConfig{},
                        WallBBoxRegistry::Ptr bbox_registry = nullptr,
                        const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter = nullptr);
    ~WallFilter() = default;

    /**
     * @brief  Voxelizza il frame, identifica i voxel-parete via RANSAC,
     *         e restituisce la voxelmap con i voxel-parete marcati.
     *
     * I voxel-parete sono marcati con is_wall = true nel DynamicGaussianVoxel.
     * DynamicObjectRejectionCPU deve saltarli nel loop di dynamic scoring
     * (vedi dynamic_object_recognition).
     *
     * @param frame  Frame preprocessato in ingresso.
     * @return       WallFilterResult con voxelmap + metadati.
     */
    WallFilterResult filter(const PreprocessedFrame& frame);

private:
    std::vector<Eigen::Vector3d> extract_centroids(
        const gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox) const;

    PlaneModel fit_plane(const Eigen::Vector3d& p0,
                         const Eigen::Vector3d& p1,
                         const Eigen::Vector3d& p2) const;

    std::vector<int> find_inliers(const std::vector<Eigen::Vector3d>& pts,
                                   const PlaneModel& plane,
                                   double threshold) const;

    PlaneModel refit_plane(const std::vector<Eigen::Vector3d>& pts,
                            const std::vector<int>& inlier_idx) const;

    /// Esegue un'iterazione RANSAC; rimuove gli inlier da pts.
    /// Restituisce false se non trova un piano con abbastanza inlier.
    bool ransac_once(std::vector<Eigen::Vector3d>& pts, PlaneModel& best_plane);

    bool is_wall_plane(const PlaneModel& plane) const;

    bool is_floor_ceiling_plane(const PlaneModel& plane) const;

    /// Marca i voxel della voxelmap il cui centroide cade su uno dei piani-parete.
    /// Restituisce il numero di voxel marcati.
    int mark_wall_voxels(gtsam_points::DynamicVoxelMapCPU& voxelmap,
                          int nvox,
                          const std::vector<PlaneModel>& wall_planes) const;
                
    BoundingBox build_wall_bbox(const std::vector<Eigen::Vector3d>& inlier_pts,
                                const PlaneModel& plane) const;
    
    std::vector<Eigen::Vector3d> extract_largest_contiguous_cluster(const std::vector<Eigen::Vector3d>& inliers,
                                                                    double cluster_eps) const;

    bool is_wall_bbox_compact(const BoundingBox& bbox) const;

    double lowest_point_z(const std::vector<Eigen::Vector3d>& pts) const;

    void polar_floor_segmentation(gtsam_points::DynamicVoxelMapCPU& voxelmap,
                                   int nvox,
                                   const std::vector<Eigen::Vector3d>& centroids) const;

    WallFilterConfig  config_;
    mutable std::mt19937 rng_;
    WallBBoxRegistry::Ptr bbox_registry_;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
    Eigen::Isometry3d last_pose_ = Eigen::Isometry3d::Identity();
};

}  // namespace glim