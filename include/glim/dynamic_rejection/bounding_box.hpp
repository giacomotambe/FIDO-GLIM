#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

/// Velocity-based 2D footprint inflation parameters.
/// Shared between DynamicBBoxRejection (BBOX mode) and
/// DynamicObjectRejectionCPU (VOXEL mode) to avoid config duplication.
struct VelocityInflationParams {
    double v_fwd_k     = 3.0;   ///< Forward footprint multiplier  (half_x + speed * k)
    double v_rear_k    = 0.45;  ///< Rear footprint multiplier
    double v_lat_k     = 0.60;  ///< Lateral footprint multiplier
    double v_vert_k    = 0.20;  ///< Legacy vertical multiplier (unused by the 2D footprint filter)
    double v_min       = 0.05;  ///< Minimum XY speed [m/s] to activate ellipsoid inflation
    double v_max_speed = 5.0;   ///< Speed saturation [m/s]: inflation is capped at this speed
    double ellipse_box_cover_scale = 1.41421356237; ///< Base ellipse scale that covers the inflated bbox corners.

    /// Load values from config_bbox_rejection.json (falls back to defaults).
    static VelocityInflationParams from_config();
};

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

    void set_velocity(const Eigen::Vector3d& v) { velocity_ = v; speed_xy_ = std::hypot(v.x(), v.y()); }
    const Eigen::Vector3d& get_velocity() const { return velocity_; }
    double get_speed_xy()               const { return speed_xy_; }

    /// Returns true if `point` falls inside the inflated 2D ellipse footprint
    /// and inside the already-inflated bbox height. The footprint is shifted
    /// along XY velocity to extend more in front than behind.
    bool contains_inflated(const Eigen::Vector4d& point,
                           const VelocityInflationParams& p) const;

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
    // Velocity for ellipsoid inflation
    Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero();
    double          speed_xy_ = 0.0;
};

}  // namespace glim
