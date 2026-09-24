#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>

#include <cmath>
#include <spdlog/spdlog.h>

namespace glim {

// ------------------------------------------------------------------ helpers
Eigen::Matrix3d PoseKalmanFilter::skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<    0, -v.z(),  v.y(),
       v.z(),     0, -v.x(),
      -v.y(),  v.x(),     0;
  return S;
}

// -------------------------------------------------------------- constructor
PoseKalmanFilter::PoseKalmanFilter()
  : delta_position_(Eigen::Vector3d::Zero()),
    velocity_(Eigen::Vector3d::Zero()),
    delta_orientation_(Eigen::Quaterniond::Identity()),
    orientation_at_reset_(Eigen::Quaterniond::Identity()),
    last_slam_pose_(Eigen::Isometry3d::Identity()),
    has_last_slam_pose_(false),
    T_delta(Eigen::Isometry3d::Identity()),
    last_filtered_pose_(Eigen::Isometry3d::Identity()),
    gravity_(0.0, 0.0, -9.81)
{
  P_ = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;
  Q_ = Eigen::Matrix<double, 9, 9>::Identity() * 0.001;
  R_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.05;
  spdlog::debug("[KF] constructor done");
}

// ----------------------------------------------------------------- getters
Eigen::Isometry3d PoseKalmanFilter::getDeltaPose() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return T_delta;
}


Eigen::Vector3d PoseKalmanFilter::getVelocity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return velocity_;
}

Eigen::Isometry3d PoseKalmanFilter::getPose() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_filtered_pose_;
}

// ----------------------------------------------------------------- predict
void PoseKalmanFilter::predict(const ImuMeasurement& imu) {
  std::lock_guard<std::mutex> lock(mutex_);
  spdlog::debug("[KF] predict: dt={:.6f} acc=({:.3f},{:.3f},{:.3f}) gyro=({:.3f},{:.3f},{:.3f})",
    imu.dt, imu.acc.x(), imu.acc.y(), imu.acc.z(), imu.gyro.x(), imu.gyro.y(), imu.gyro.z());

  const double dt = imu.dt;

  // Current absolute orientation = orientation_at_reset_ * delta_orientation_
  const Eigen::Quaterniond q_abs = (orientation_at_reset_ * delta_orientation_).normalized();
  const Eigen::Matrix3d R_abs = q_abs.toRotationMatrix();

  // Accelerometer in world frame (absolute) to integrate velocity/position
  const Eigen::Vector3d acc_world = R_abs * imu.acc + gravity_;

  // But we accumulate delta in the previous-frame coordinate system.
  // Rotate acc_world into the frame at reset (= previous SLAM frame).
  const Eigen::Matrix3d R_reset = orientation_at_reset_.toRotationMatrix();
  const Eigen::Vector3d acc_local = R_reset.transpose() * acc_world;

  spdlog::debug("[KF] predict: acc_world=({:.3f},{:.3f},{:.3f}) acc_local=({:.3f},{:.3f},{:.3f})",
    acc_world.x(), acc_world.y(), acc_world.z(), acc_local.x(), acc_local.y(), acc_local.z());

  // --- Nominal-state integration (in local / previous-frame coords) ---
  delta_position_ += velocity_ * dt + 0.5 * acc_local * dt * dt;
  velocity_ += acc_local * dt;

  // Rotation delta integration
  const double angle = imu.gyro.norm() * dt;
  if (angle > 1e-12) {
    const Eigen::Vector3d axis = imu.gyro.normalized();
    delta_orientation_ = (delta_orientation_ * Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis))).normalized();
  }
  T_delta = Eigen::Isometry3d::Identity();
  T_delta.linear() = delta_orientation_.toRotationMatrix();
  T_delta.translation() = delta_position_;

  last_filtered_pose_ = last_slam_pose_ * T_delta;
  // --- Error-state transition Jacobian F (9×9) ---
  const Eigen::Matrix3d R_delta = delta_orientation_.toRotationMatrix();
  Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;                   // ∂δp / ∂δv
  F.block<3, 3>(3, 6) = -R_delta * skew(imu.acc) * dt;                      // ∂δv / ∂δθ
  F.block<3, 3>(6, 6) -= skew(imu.gyro) * dt;                               // ∂δθ / ∂δθ

  P_ = F * P_ * F.transpose() + Q_;

  spdlog::debug("[KF] predict done: dp=({:.4f},{:.4f},{:.4f}) vel=({:.4f},{:.4f},{:.4f})",
    delta_position_.x(), delta_position_.y(), delta_position_.z(),
    velocity_.x(), velocity_.y(), velocity_.z());
}

// ----------------------------------------------------------------- update
Eigen::Isometry3d PoseKalmanFilter::update(const Eigen::Isometry3d& T_world_imu) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Eigen::Vector3d t = T_world_imu.translation();
  spdlog::debug("[KF] update: T_world_imu t=({:.4f},{:.4f},{:.4f}) has_last={}", t.x(), t.y(), t.z(), has_last_slam_pose_);

  if (!has_last_slam_pose_) {
    // First SLAM pose: just store it, return identity (no motion yet)
    last_slam_pose_ = T_world_imu;
    has_last_slam_pose_ = true;
    orientation_at_reset_ = Eigen::Quaterniond(T_world_imu.rotation()).normalized();

    // Reset the accumulated state
    delta_position_.setZero();
    velocity_.setZero();
    delta_orientation_ = Eigen::Quaterniond::Identity();
    P_ = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;

    spdlog::debug("[KF] update: first pose stored, returning T_world_imu");
    return T_world_imu;
  }

  // --- SLAM-observed relative transform (in previous-frame coords) ---
  spdlog::debug("[KF] update: computing SLAM delta");
  const Eigen::Isometry3d T_slam_delta = last_slam_pose_.inverse() * T_world_imu;

  const Eigen::Vector3d slam_dp = T_slam_delta.translation();
  Eigen::Quaterniond slam_dq(T_slam_delta.rotation());
  slam_dq.normalize();

  spdlog::debug("[KF] update: slam_dp=({:.4f},{:.4f},{:.4f}) imu_dp=({:.4f},{:.4f},{:.4f})",
    slam_dp.x(), slam_dp.y(), slam_dp.z(), delta_position_.x(), delta_position_.y(), delta_position_.z());

  // --- Innovation (SLAM delta - IMU-predicted delta) ---
  const Eigen::Vector3d y_p = slam_dp - delta_position_;

  Eigen::Quaterniond q_pred = delta_orientation_;
  if (slam_dq.dot(q_pred) < 0.0) {
    slam_dq.coeffs() = -slam_dq.coeffs();
  }
  const Eigen::Quaterniond q_err = slam_dq * q_pred.inverse();
  const Eigen::Vector3d y_theta = 2.0 * q_err.vec();

  spdlog::debug("[KF] update: innovation y_p=({:.4f},{:.4f},{:.4f}) y_theta=({:.4f},{:.4f},{:.4f})",
    y_p.x(), y_p.y(), y_p.z(), y_theta.x(), y_theta.y(), y_theta.z());

  Eigen::Matrix<double, 6, 1> y;
  y.head<3>() = y_p;
  y.tail<3>() = y_theta;

  // --- Observation Jacobian H (6×9) ---
  Eigen::Matrix<double, 6, 9> H = Eigen::Matrix<double, 6, 9>::Zero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

  // --- Kalman gain ---
  spdlog::debug("[KF] update: computing Kalman gain");
  const Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_;
  const Eigen::Matrix<double, 9, 6> K = P_ * H.transpose() * S.inverse();

  // --- Apply correction ---
  const Eigen::Matrix<double, 9, 1> dx = K * y;
  spdlog::debug("[KF] update: correction dx_p=({:.4f},{:.4f},{:.4f}) dx_v=({:.4f},{:.4f},{:.4f}) dx_th=({:.4f},{:.4f},{:.4f})",
    dx(0), dx(1), dx(2), dx(3), dx(4), dx(5), dx(6), dx(7), dx(8));

  delta_position_ += dx.segment<3>(0);
  velocity_ += dx.segment<3>(3);

  const Eigen::Vector3d dtheta = dx.segment<3>(6);
  if (dtheta.norm() > 1e-12) {
    const Eigen::Vector3d axis = dtheta.normalized();
    Eigen::Quaterniond dq(Eigen::AngleAxisd(dtheta.norm(), axis));
    delta_orientation_ = (dq * delta_orientation_).normalized();
  }

  // --- Build filtered relative transform ---
  T_delta = Eigen::Isometry3d::Identity();
  T_delta.translation() = delta_position_;
  T_delta.linear() = delta_orientation_.toRotationMatrix();

  // --- Compute filtered absolute pose: T_world_imu = last_slam_pose * T_delta ---
  const Eigen::Isometry3d T_filtered = last_slam_pose_ * T_delta;

  spdlog::debug("[KF] update: filtered abs t=({:.4f},{:.4f},{:.4f})", T_filtered.translation().x(), T_filtered.translation().y(), T_filtered.translation().z());

  // --- Reset for next interval ---
  // velocity_ is expressed in the frame at reset: carry it over to the new reset frame
  // instead of zeroing it, otherwise the IMU prediction restarts from rest every update.
  const Eigen::Vector3d v_world = orientation_at_reset_ * velocity_;
  last_slam_pose_ = T_world_imu;
  orientation_at_reset_ = Eigen::Quaterniond(T_world_imu.rotation()).normalized();
  delta_position_.setZero();
  velocity_ = orientation_at_reset_.conjugate() * v_world;
  delta_orientation_ = Eigen::Quaterniond::Identity();

  const Eigen::Matrix<double, 9, 9> I9 = Eigen::Matrix<double, 9, 9>::Identity();
  P_ = (I9 - K * H) * P_;

  spdlog::debug("[KF] update: reset done, returning T_filtered");
  last_filtered_pose_ = T_filtered;
  return T_filtered;
}

}  // namespace glim