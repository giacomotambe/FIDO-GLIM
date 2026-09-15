# Configuration reference

All FIDO parameters below share the same load pattern: `Config config(GlobalConfig::get_config_path("config_<name>"))`, then `config.param<T>("section", "key", default)` — any key omitted from the JSON silently falls back to the C++ default listed in the corresponding header. Comments (`//`, `/* */`) are stripped by `glim::Config` before parsing, so every shipped file documents itself inline.

The five files below are FIDO-specific; the rest (`config_odometry_*`, `config_sub_mapping_*`, `config_viewer.json`, …) are inherited from upstream GLIM and untouched here.

## `config_wall_filter.json`

Read by [`WallFilter`](module1_voxel.md#2-static-pre-filtering-wallfilter).

| Key | Default |
|---|---|
| `voxel_resolution` | 0.25 m |
| `ransac_max_iterations` / `ransac_min_inliers` / `ransac_confidence` | 100 / 30 / 0.99 |
| `ransac_inlier_threshold` / `floor_ransac_inlier_threshold` | 0.07 m / 0.15 m |
| `wall_vertical_angle_deg` / `floor_ceiling_angle_deg` | 5.0° |
| `max_planes` | 8 |
| `wall_bbox_max_aspect_ratio` | 5.0 |
| `floor_polar_enabled` | true |
| `floor_polar_r_bin` / `floor_polar_theta_deg` | 0.5 m / 10.0° |
| `floor_polar_seed_min_r` / `_seed_max_r` | 2.0 m / 15.0 m |
| `floor_polar_k_lowest` | 8 |
| `floor_polar_flatness_threshold` / `_slope_threshold` | 0.05 / 0.1 |
| `floor_polar_max_ground_height` / `_max_vertical_spread` | 0.3 m / 0.30 m |

## `config_wall_registry.json`

Read by `WallBBoxRegistry`.

| Key | Default |
|---|---|
| `overlap_threshold` / `merge_weight` | 0.3 / 0.3 |
| `enable_expiry` / `max_missed_frames` | true / 5 |
| `min_normal_dot` | 0.9 |
| `max_center_distance` / `max_duplicate_center_distance` | 5.0 m / 2.0 m |
| `max_wall_thickness` | 0.9 m |

## `config_dynamic_cluster_extractor.json`

Read by [`DynamicClusterExtractor`](module1_voxel.md#3-cluster-extraction-tracking-dynamicclusterextractor).

| Key | Default |
|---|---|
| `eps_voxel_factor` / `min_pts` / `knn_max_neighbors` | 1.5 / 2 / 32 |
| `min_cluster_voxels` / `min_points_for_bbox` | 3 / 3 |
| `bbox_min_extent` / `_max_extent` | 0.1 m / 4.0 m |
| `bbox_min_volume` / `_max_volume` | 0.0001 m³ / 1000.0 m³ |
| `cluster_iou_threshold` | 0.1 |
| `peer_merge_distance_indoor` / `_outdoor` | 0.01 m / 2.8 m |
| `track_match_distance` / `track_match_iou` | 2.0 m / 0.05 |
| `track_max_missed` | 5 frames |
| `min_dynamic_frames` | 2 |
| `permanent_dynamic_frames` / `_static_frames` | 8 / 20 |
| `track_bbox_history_size` | 5 |

## `config_dynamic_object_rejection.json`

Read by [`DynamicObjectRejectionCPU`](module1_voxel.md#4-dynamic-scoring-dynamicobjectrejectioncpu).

| Key | Default |
|---|---|
| `dynamic_score_threshold` | 0.08 |
| `tier1_threshold_factor` / `memory_threshold_factor` / `unconstrained_threshold_factor` | 0.55 / 0.70 / 3.0 |
| `w_shift` / `w_mahalanobis` / `w_neighbor` | 1.2 / 0.0 / 0.2 |
| `w_cluster` / `w_history` / `w_history_dynamic` | 0.0 / 0.3 / 0.2 |
| `w_distance` / `w_velocity` / `velocity_static_threshold` | 0.16 / 0.0 / 0.0 |
| `frame_num_memory` / `history_factor` | 5 / 0.3 |
| `min_shift_m` | 0.03 m |
| `cluster_propagation_threshold` | 0.3 |
| `motion_threshold_scale` / `rotation_threshold_scale` | 0.8 / 2.2 |
| `cluster_motion_scale` / `cluster_rotation_scale` | 2.0 / 3.0 |
| `points_limit` | 0.05 |
| `num_threads` | 4 |

## `config_bbox_rejection.json`

Read by [`DynamicBBoxRejection`](module2_bbox.md) and shared `VelocityInflationParams`.

| Key | Default |
|---|---|
| `inflate_margin` | 0.4 m |
| `max_bbox_frames` | 10 |
| `velocity_inflation_k` / `reverse_velocity_inflation_k` | 0.85 / 0.35 |
| `rear_inflation_k` <span class="unused">(shadowed — see [Module 2](module2_bbox.md#velocity-inflated-footprint))</span> | 0.15 |
| `lateral_inflation_k` / `vertical_inflation_k` | 0.30 / 0.20 |
| `velocity_inflation_min` / `_max_speed` | 0.1 m/s / 3.0 m/s |
| `ellipse_box_cover_scale` | 1.41421356237 |
| `bbox_potentially_dynamic_min_track_age` <span class="unused">(unused)</span> | 5 |
| `bbox_static_min_track_age` <span class="unused">(unused)</span> | 15 |
| `bbox_min_static_velocity` <span class="unused">(unused)</span> | 0.1 |
| `bbox_rejection_obstacle_min_removed_points` <span class="unused">(unused)</span> | 3 |

## `config_ros.json` — FIDO-relevant keys

| Key | Values | Effect |
|---|---|---|
| `dynamic_rejection_type` | `NONE` · `CLUSTER` · `BBOX` · `COMBINED` | Selects which module(s) run for the session |
| `env_type` | `INDOOR` · `OUTDOOR` | Chooses the peer-merge distance in `DynamicClusterExtractor` |
| `bbox_topic` | `/onboard_detector/tracked_dynamic_obstacles` | External detector boxes for Module 2 (shipped default) |
| `imu_topic` | `/imu/data` | IMU input |
| `points_topic` | `/velodyne_points` | LiDAR input |
