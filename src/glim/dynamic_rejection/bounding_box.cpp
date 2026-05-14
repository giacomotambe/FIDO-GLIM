
#include <vector>
#include <Eigen/Core>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <cmath>
#include <glim/util/config.hpp>

namespace glim {

VelocityInflationParams VelocityInflationParams::from_config() {
    VelocityInflationParams p;
    Config config(GlobalConfig::get_config_path("config_bbox_rejection"));
    p.v_fwd_k     = config.param<double>("param_bbox_rejection", "velocity_inflation_k",        p.v_fwd_k);
    p.v_rear_k    = config.param<double>("param_bbox_rejection", "rear_inflation_k",             p.v_rear_k);
    p.v_rear_k    = config.param<double>("param_bbox_rejection", "reverse_velocity_inflation_k", p.v_rear_k);
    p.v_lat_k     = config.param<double>("param_bbox_rejection", "lateral_inflation_k",          p.v_lat_k);
    p.v_vert_k    = config.param<double>("param_bbox_rejection", "vertical_inflation_k",         p.v_vert_k);
    p.v_min       = config.param<double>("param_bbox_rejection", "velocity_inflation_min",       p.v_min);
    p.v_max_speed = config.param<double>("param_bbox_rejection", "velocity_inflation_max_speed", p.v_max_speed);
    p.ellipse_box_cover_scale =
        config.param<double>("param_bbox_rejection", "ellipse_box_cover_scale", p.ellipse_box_cover_scale);
    return p;
}

BoundingBox::BoundingBox()
    : size(Eigen::Vector3d::Zero()),
      center(Eigen::Vector3d::Zero()),
      rotation(Eigen::Matrix3d::Identity()),
      is_dynamic(false),
      is_locked_(false),
      track_id(-1),
      R_inv(Eigen::Matrix3d::Identity()),
      half_size(Eigen::Vector3d::Zero())
{}

BoundingBox::BoundingBox(const Eigen::Vector3d& size,
                         const Eigen::Vector3d& center,
                         const Eigen::Matrix3d& rotation)
    : size(size),
      center(center),
      rotation(rotation),
      is_dynamic(false),
      is_locked_(false),
      track_id(-1)
{
    // Precompute values used in contains()
    R_inv = rotation.transpose();
    half_size = size * 0.5;
}


bool BoundingBox::contains(const Eigen::Vector4d& point) const {
    // Estrarre solo le coordinate xyz
    Eigen::Vector3d p = point.head<3>();

    // Trasformare il punto nel frame locale della bounding box
    Eigen::Vector3d local_p = R_inv * (p - center);

    // Controllo limiti
    return (std::abs(local_p.x()) <= half_size.x() &&
            std::abs(local_p.y()) <= half_size.y() &&
            std::abs(local_p.z()) <= half_size.z());
}


bool BoundingBox::contains_bbox(const BoundingBox& inner) const {
    // AABB containment: inner is fully inside this if its interval is a subset on all axes.
    const Eigen::Vector3d this_min  = center - half_size;
    const Eigen::Vector3d this_max  = center + half_size;
    const Eigen::Vector3d inner_min = inner.center - inner.half_size;
    const Eigen::Vector3d inner_max = inner.center + inner.half_size;
    return (inner_min.x() >= this_min.x() && inner_max.x() <= this_max.x() &&
            inner_min.y() >= this_min.y() && inner_max.y() <= this_max.y() &&
            inner_min.z() >= this_min.z() && inner_max.z() <= this_max.z());
}

void BoundingBox::transform(const Eigen::Isometry3d& T) {
    // Aggiorna centro e rotazione
    center = T * center;
    rotation = T.linear() * rotation;

    // Aggiorna la matrice inversa per contains()
    R_inv = rotation.transpose();
}




void BoundingBox::inflate(double margin) {
    size += Eigen::Vector3d::Constant(2.0 * margin);
    half_size = size * 0.5;
}

double BoundingBox::iou(const BoundingBox& other) const {
    // Intersezione: clamp dei bound sovrapposti
    const Eigen::Vector3d inter_min = (center - half_size).cwiseMax(other.center - other.half_size);
    const Eigen::Vector3d inter_max = (center + half_size).cwiseMin(other.center + other.half_size);
    const Eigen::Vector3d inter_size = (inter_max - inter_min).cwiseMax(Eigen::Vector3d::Zero());

    const double vol_inter = inter_size.x() * inter_size.y() * inter_size.z();
    if (vol_inter <= 0.0) return 0.0;

    const double vol_a = (2.0 * half_size).prod();
    const double vol_b = (2.0 * other.half_size).prod();

    return vol_inter / (vol_a + vol_b - vol_inter + 1e-9);
}

bool BoundingBox::contains_inflated(const Eigen::Vector4d& point,
                                     const VelocityInflationParams& p) const
{
    // Clamp effective speed to avoid unbounded ellipsoid growth at high speed.
    const double eff_speed = speed_xy_ > p.v_min ? std::min(speed_xy_, p.v_max_speed) : 0.0;

    const Eigen::Vector3d p3 = point.head<3>();
    const Eigen::Vector3d dp = p3 - center;
    const Eigen::Vector3d local_p = R_inv * dp;

    if (std::abs(local_p.z()) > half_size.z()) {
        return false;
    }

    double cos_h = rotation(0, 0);
    double sin_h = rotation(1, 0);
    if (speed_xy_ > p.v_min) {
        cos_h = velocity_.x() / speed_xy_;
        sin_h = velocity_.y() / speed_xy_;
    }

    // Decompose displacement into heading / lateral / vertical axes
    const double center_shift = 0.5 * eff_speed * (p.v_fwd_k - p.v_rear_k);
    const double shifted_x = dp.x() - center_shift * cos_h;
    const double shifted_y = dp.y() - center_shift * sin_h;
    const double u =  shifted_x * cos_h + shifted_y * sin_h;   // along heading
    const double v = -shifted_x * sin_h + shifted_y * cos_h;   // lateral

    const double cover_scale = std::max(p.ellipse_box_cover_scale, 1.0);
    const Eigen::Vector2d heading_axis(cos_h, sin_h);
    const Eigen::Vector2d lateral_axis(-sin_h, cos_h);
    const Eigen::Vector2d box_x_axis(rotation(0, 0), rotation(1, 0));
    const Eigen::Vector2d box_y_axis(rotation(0, 1), rotation(1, 1));
    const double bbox_half_u =
        std::abs(heading_axis.dot(box_x_axis)) * half_size.x() +
        std::abs(heading_axis.dot(box_y_axis)) * half_size.y();
    const double bbox_half_v =
        std::abs(lateral_axis.dot(box_x_axis)) * half_size.x() +
        std::abs(lateral_axis.dot(box_y_axis)) * half_size.y();
    const double semi_u = std::max(
        cover_scale * bbox_half_u + 0.5 * eff_speed * (p.v_fwd_k + p.v_rear_k),
        1e-6);
    const double semi_lat = std::max(cover_scale * bbox_half_v + eff_speed * p.v_lat_k, 1e-6);

    const double eu = u / semi_u;
    const double ev = v / semi_lat;

    return (eu * eu + ev * ev <= 1.0);
}

double BoundingBox::overlap(const BoundingBox& other) const {
    const Eigen::Vector3d inter_min = (center - half_size).cwiseMax(other.center - other.half_size);
    const Eigen::Vector3d inter_max = (center + half_size).cwiseMin(other.center + other.half_size);
    const Eigen::Vector3d inter_size = (inter_max - inter_min).cwiseMax(Eigen::Vector3d::Zero());

    const double vol_inter = inter_size.x() * inter_size.y() * inter_size.z();
    if (vol_inter <= 0.0) return 0.0;

    const double vol_a = (2.0 * half_size).prod();
    const double vol_b = (2.0 * other.half_size).prod();
    const double vol_min = std::min(vol_a, vol_b);
    if (vol_min <= 1e-9) return 0.0;

    return vol_inter / vol_min;
}

}  // namespace glim
