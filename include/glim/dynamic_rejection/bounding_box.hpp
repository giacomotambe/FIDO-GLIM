#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

class BoundingBox {
public:
    BoundingBox();
    BoundingBox(const Eigen::Vector3d& size,
                const Eigen::Vector3d& center,
                const Eigen::Matrix3d& rotation);

    ~BoundingBox() = default;

    bool contains(const Eigen::Vector4d& point) const;
    /// Returns true if `inner` is fully contained inside this bbox (AABB check).
    bool contains_bbox(const BoundingBox& inner) const;
    void transform(const Eigen::Isometry3d& T);
    /// Expands the bounding box by `margin` in all directions (adds margin on each side per axis).
    void inflate(double margin);
    double iou(const BoundingBox& other) const;
    /// Overlap coefficient: intersection / min(vol_a, vol_b).
    /// Returns 1.0 when the smaller box is fully inside the larger one.
    double overlap(const BoundingBox& other) const;
    // -----------------------------------------------------------------------
    // Getters (needed by WallBBoxRegistry for IoU / merge operations)
    // -----------------------------------------------------------------------
    const Eigen::Vector3d& get_size()     const { return size; }
    const Eigen::Vector3d& get_center()   const { return center; }
    const Eigen::Matrix3d& get_rotation() const { return rotation; }
    const bool is_dynamic_bbox() const { return is_dynamic; }
    void set_dynamic(bool dynamic) { is_dynamic = dynamic; }

    /// A locked bbox has reached a permanent state and propagate_to_clusters()
    /// must not override its is_dynamic flag.
    bool is_locked()         const { return is_locked_; }
    void set_locked(bool v)        { is_locked_ = v; }

    int  get_track_id() const { return track_id; }
    void set_track_id(int id)  { track_id = id; }

private:
    Eigen::Vector3d size;
    Eigen::Vector3d center;
    Eigen::Matrix3d rotation;
    bool is_dynamic;
    bool is_locked_ = false;  ///< True when the track has reached a permanent dynamic/static state.
    int  track_id;  ///< -1 = untracked / phantom
    // Precomputed for contains()
    Eigen::Matrix3d R_inv;
    Eigen::Vector3d half_size;
};

}  // namespace glim