#pragma once

#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

struct ImuMeasurement {
  Eigen::Vector3d acc;
  Eigen::Vector3d gyro;
  double dt;
};

/**
 * @brief Error-State Kalman Filter that estimates the absolute pose
 *        T_world_imu by fusing IMU predictions with SLAM updates.
 *
 * Between SLAM updates the filter integrates IMU measurements to accumulate
 * the incremental motion (position delta + rotation delta + velocity).
 *
 * When a new SLAM pose arrives, update() computes the SLAM-observed delta,
 * fuses it with the IMU prediction, and returns the filtered absolute pose
 * T_world_imu.
 *
 * Nominal state (relative to previous frame):
 *   Δp (3)  – position delta
 *   Δv (3)  – velocity at current time
 *   Δq (4)  – rotation delta (quaternion)
 *
 * Error state (9-dim): δp (3), δv (3), δθ (3)
 */
class PoseKalmanFilter {
public:
  PoseKalmanFilter();

  /// Return the current accumulated relative transform (T_{k-1 -> k})
  Eigen::Isometry3d getDeltaPose() const;
  
  Eigen::Isometry3d getPose() const;


  /// Return the current velocity estimate
  Eigen::Vector3d getVelocity() const;

  /// IMU prediction: accumulate relative motion
  void predict(const ImuMeasurement& imu);

  /**
   * @brief SLAM update: fuse the SLAM-observed absolute pose.
   *
   * Internally computes the SLAM-measured delta from the previous SLAM pose,
   * fuses with IMU-integrated delta, resets for next interval, and returns
   * the filtered absolute pose T_world_imu.
   *
   * @param T_world_imu  Absolute SLAM pose at current time
   * @return Filtered absolute pose T_world_imu
   */
  Eigen::Isometry3d update(const Eigen::Isometry3d& T_world_imu);

  /**
   * @brief SLAM update with the timestamp of the measured pose.
   *
   * Same as update(T_world_imu); the last two stamped poses are also kept to predict the pose of
   * the scan being processed by the dynamic rejection when there is no IMU (see getFramePose()).
   */
  Eigen::Isometry3d update(const Eigen::Isometry3d& T_world_imu, double stamp);

  /// Stamp of the scan the dynamic rejection is about to process.
  void setFrameStamp(double stamp);

  /**
   * @brief Pose of the scan set with setFrameStamp(), used by the dynamic rejection.
   *
   * With IMU predictions (or before two stamped SLAM poses are available) this is getPose().
   * Without IMU, getPose() stays at the last SLAM pose, which is one or more scans old when the
   * rejection runs; the pose is instead extrapolated at constant velocity from the last two
   * stamped SLAM poses to the scan stamp (at most max_extrapolation_ seconds ahead).
   */
  Eigen::Isometry3d getFramePose() const;

private:
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v);

  // Nominal relative state (accumulated since last reset)
  Eigen::Vector3d delta_position_;
  Eigen::Vector3d velocity_;
  Eigen::Quaterniond delta_orientation_;
  Eigen::Isometry3d last_filtered_pose_;  // T_world_imu at last SLAM update (for delta computation)
  Eigen::Isometry3d T_delta;

  // Orientation at the start of the current interval (for rotating acc to local-level frame)
  Eigen::Quaterniond orientation_at_reset_;

  // Previous SLAM pose (to compute SLAM-observed delta)
  Eigen::Isometry3d last_slam_pose_;
  bool has_last_slam_pose_;

  // Error-state covariance (9×9: δp, δv, δθ)
  Eigen::Matrix<double, 9, 9> P_;
  Eigen::Matrix<double, 9, 9> Q_;   // process noise
  Eigen::Matrix<double, 6, 6> R_;   // measurement noise (pos 3 + rot 3)

  Eigen::Vector3d gravity_;

  // Constant-velocity prediction of the scan pose when there is no IMU
  bool imu_used_ = false;                  // set by predict()
  double frame_stamp_ = -1.0;              // stamp of the scan being processed (setFrameStamp)
  int num_stamped_ = 0;                    // number of valid entries in stamped_poses_ (0..2)
  double stamps_[2] = {0.0, 0.0};          // [0] older, [1] newer
  Eigen::Isometry3d stamped_poses_[2] = {Eigen::Isometry3d::Identity(), Eigen::Isometry3d::Identity()};
  static constexpr double max_extrapolation_ = 1.0;  // [s]

  mutable std::mutex mutex_;  // protects all state from concurrent access
};

}  // namespace glim
