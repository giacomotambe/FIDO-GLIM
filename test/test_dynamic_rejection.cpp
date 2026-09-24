// Minimal self-contained tests for the dynamic rejection module.
// Build: cmake -DBUILD_DYNAMIC_REJECTION_TESTS=ON ..  &&  ./test_dynamic_rejection
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Geometry>
#include <glim/dynamic_rejection/bounding_box.hpp>

using namespace glim;

static int failures = 0;
#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
  } while (0)

static BoundingBox unit_box() {
  return BoundingBox(Eigen::Vector3d(1.0, 1.0, 2.0), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
}

static void test_contains() {
  const auto b = unit_box();
  CHECK(b.contains(Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)));
  CHECK(b.contains(Eigen::Vector4d(0.45, -0.45, 0.9, 1.0)));
  CHECK(!b.contains(Eigen::Vector4d(0.6, 0.0, 0.0, 1.0)));
  CHECK(!b.contains(Eigen::Vector4d(0.0, 0.0, 1.1, 1.0)));
}

static void test_contains_inflated() {
  VelocityInflationParams p;  // defaults: v_fwd_k=3.0, v_rear_k=0.45
  auto b = unit_box();

  // Static bbox: no velocity-driven elongation far ahead.
  CHECK(!b.contains_inflated(Eigen::Vector4d(3.0, 0.0, 0.0, 1.0), p));

  // Moving at 2 m/s along +x: footprint extends forward much more than backward.
  b.set_velocity(Eigen::Vector3d(2.0, 0.0, 0.0));
  CHECK(b.contains_inflated(Eigen::Vector4d(2.0, 0.0, 0.0, 1.0), p));
  CHECK(!b.contains_inflated(Eigen::Vector4d(-2.0, 0.0, 0.0, 1.0), p));
  // Height is never inflated.
  CHECK(!b.contains_inflated(Eigen::Vector4d(0.5, 0.0, 1.5, 1.0), p));
}

static void test_flags() {
  auto b = unit_box();
  CHECK(!b.has_strong_motion());
  b.set_strong_motion(true);
  CHECK(b.has_strong_motion());
  b.set_dynamic(true);
  b.set_locked(true);
  CHECK(b.is_dynamic_bbox() && b.is_locked());
}

// Documents the frame convention used in DynamicObjectRejectionCPU::reject() and
// DynamicClusterExtractor::extract_clusters(): with poses T_world_sensor, a point in
// the current sensor frame maps to the previous sensor frame with T_prev^-1 * T_cur.
static void test_pose_convention() {
  Eigen::Isometry3d T_prev = Eigen::Isometry3d::Identity();
  T_prev.linear() = Eigen::AngleAxisd(0.8, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_prev.translation() = Eigen::Vector3d(10.0, -4.0, 0.5);

  Eigen::Isometry3d T_cur = Eigen::Isometry3d::Identity();
  T_cur.linear() = Eigen::AngleAxisd(1.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_cur.translation() = Eigen::Vector3d(11.0, -3.5, 0.5);

  const Eigen::Vector3d p_cur(3.0, 1.0, 0.2);
  const Eigen::Vector3d p_prev_truth = T_prev.inverse() * (T_cur * p_cur);

  const Eigen::Isometry3d T_new = T_prev.inverse() * T_cur;   // fixed
  const Eigen::Isometry3d T_old = T_cur * T_prev.inverse();   // previous code
  CHECK((T_new * p_cur - p_prev_truth).norm() < 1e-9);
  CHECK((T_old * p_cur - p_prev_truth).norm() > 1.0);          // old one is wrong far from origin

  // Tracker: previous-frame quantities -> current frame.
  const Eigen::Isometry3d T_to_current = T_cur.inverse() * T_prev;
  CHECK((T_to_current * p_prev_truth - p_cur).norm() < 1e-9);
}

int main() {
  test_contains();
  test_contains_inflated();
  test_flags();
  test_pose_convention();
  if (failures == 0) std::printf("All dynamic rejection tests passed\n");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
