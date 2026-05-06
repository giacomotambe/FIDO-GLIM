
#include <vector>
#include <Eigen/Core>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <cmath>

namespace glim {

BoundingBox::BoundingBox()
    : size(Eigen::Vector3d::Zero()),
      center(Eigen::Vector3d::Zero()),
      rotation(Eigen::Matrix3d::Identity()),
      is_dynamic(false),
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
      is_dynamic(false),  // default to static, can be set later
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