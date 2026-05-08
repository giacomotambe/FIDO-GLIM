#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>

namespace gtsam_points {
// Forward declaration
class DynamicVoxelMapCPU;

struct DynamicGaussianVoxel : public gtsam_points::GaussianVoxel
{
    public:
        using Ptr = std::shared_ptr<DynamicGaussianVoxel>;
        using ConstPtr = std::shared_ptr<const DynamicGaussianVoxel>;

        DynamicGaussianVoxel() : gtsam_points::GaussianVoxel(), is_dynamic(false), is_wall(false), is_ground(false), is_outlier(false), voxel_point_cloud(nullptr), dynamic_score(0.0) {}
    public:
        bool is_dynamic;
        bool is_wall;
        bool is_ground; 
        bool is_outlier;
        std::vector<Eigen::Vector4d> voxel_points; 
        std::vector<double> voxel_intensities;
        std::vector<double> voxel_times;
        PointCloudCPU::Ptr voxel_point_cloud; // store the history of this voxel for dynamic classification
        double dynamic_score; // store the dynamic score for this voxel


        // override must use fully qualified Setting type
        void add(const gtsam_points::GaussianVoxel::Setting& setting, const gtsam_points::PointCloud& points, size_t i);    
};


class DynamicVoxelMapCPU : public gtsam_points::GaussianVoxelMap , public gtsam_points::IncrementalVoxelMap<DynamicGaussianVoxel> {
    public:
        using Ptr = std::shared_ptr<DynamicVoxelMapCPU>;
        using ConstPtr = std::shared_ptr<const DynamicVoxelMapCPU>;

        DynamicVoxelMapCPU(double resolution);
        virtual ~DynamicVoxelMapCPU() override {}

        /**
         * @brief Look up a voxel by its ID (modifiable)
         * @param voxel_id The ID of the voxel to look up
         * @return Reference to the DynamicGaussianVoxel
         */
        DynamicGaussianVoxel& lookup_voxel(int voxel_id);

        /**
         * @brief Look up a voxel by its ID (const)
         * @param voxel_id The ID of the voxel to look up
         * @return Const reference to the DynamicGaussianVoxel
         */
        const DynamicGaussianVoxel& lookup_voxel(int voxel_id) const;

        /**
         * @brief Insert a point cloud frame into the voxel map
         * @param frame The point cloud frame to insert
         */
        virtual double voxel_resolution() const override;

        /// @brief Compute the voxel index corresponding to a point.
        Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;

        /// @brief Look up a voxel index. If the voxel does not exist, return -1.
        int lookup_voxel_index(const Eigen::Vector3i& coord) const;

        /// @brief  Insert a point cloud frame into the voxelmap.
        virtual void insert(const gtsam_points::PointCloud& frame) override;

        /// @brief Extract all individual points from all voxels (not just means)
        gtsam_points::PointCloudCPU::Ptr all_points_data() const;

        /// @brief Save the voxelmap to a compact binary format
        virtual void save_compact(const std::string& path) const override;

      


};

namespace frame {
template <>
struct traits<DynamicGaussianVoxel> {
  static int size(const DynamicGaussianVoxel& frame) { return frame.size(); }

  static bool has_points(const DynamicGaussianVoxel& frame) { return true; }
  static bool has_normals(const DynamicGaussianVoxel& frame) { return false; }
  static bool has_covs(const DynamicGaussianVoxel& frame) { return true; }
  static bool has_intensities(const DynamicGaussianVoxel& frame) { return true; }

  static const Eigen::Vector4d& point(const DynamicGaussianVoxel& frame, size_t i) { return frame.mean; }
  static const Eigen::Vector4d normal(const DynamicGaussianVoxel& frame, size_t i) { return Eigen::Vector4d::Zero(); }
  static const Eigen::Matrix4d& cov(const DynamicGaussianVoxel& frame, size_t i) { return frame.cov; }
  static double intensity(const DynamicGaussianVoxel& frame, size_t i) { return frame.intensity; }
};
}

namespace frame {
template <>
struct traits<DynamicVoxelMapCPU> {
  static bool has_points(const DynamicVoxelMapCPU& ivox) { return ivox.has_points(); }
  static bool has_normals(const DynamicVoxelMapCPU& ivox) { return ivox.has_normals(); }
  static bool has_covs(const DynamicVoxelMapCPU& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const DynamicVoxelMapCPU& ivox) { return ivox.has_intensities(); }

  static decltype(auto) point(const DynamicVoxelMapCPU& ivox, size_t i) { return ivox.point(i); }
  static decltype(auto) normal(const DynamicVoxelMapCPU& ivox, size_t i) { return ivox.normal(i); }
  static decltype(auto) cov(const DynamicVoxelMapCPU& ivox, size_t i) { return ivox.cov(i); }
  static decltype(auto) intensity(const DynamicVoxelMapCPU& ivox, size_t i) { return ivox.intensity(i); }
};
}

}  // namespace gtsam_points





