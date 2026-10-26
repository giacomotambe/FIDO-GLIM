# Configuration reference

All FIDO parameters share the same load pattern: `Config config(GlobalConfig::get_config_path("config_<name>"))`, then `config.param<T>("section", "key", default)`. A key missing from the JSON silently falls back to the C++ default, which often differs from the shipped value — the tables below list the **shipped** value. Comments (`//`, `/* */`) are stripped by `glim::Config`.

Status: **active** — used with effect at the shipped value · **option** — off by default, working · **legacy** — read but without effect (kept for old configs).

For how to tune them, see [Tuning guide](tuning.md).

## Before you start

| Check | Where |
|---|---|
| `dynamic_rejection_type` is not `"NONE"` — otherwise the preprocessor does not keep the raw scan and the rejection falls back to the odometry frame | `config_ros.json` |
| `apply_lidar_imu_extrinsic` is `true` only if `T_lidar_imu` in `config_sensors.json` belongs to *your* sensor; with poses already in the LiDAR frame use `false` (both files) | `config_dynamic_object_rejection.json`, `config_dynamic_cluster_extractor.json` |
| The pose of the current scan is predicted: IMU, or stamped pose updates (constant-velocity prediction) | [Module 1, step 0](module1_voxel.md#0-pose-of-the-scan-posekalmanfilter) |
| Without IMU, use the CT odometry (`config_odometry_ct.json`) | `config.json` |

## `config_dynamic_object_rejection.json`

Section `dynamic_object_rejection`, read by [`DynamicObjectRejectionCPU`](module1_voxel.md#4-voxel-evidence-dynamicobjectrejectioncpu).

### Input and density

| Key | Default | Meaning |
|---|---|---|
| `rej_input_points` | 10000 | Uniform sample of the raw scan used as rejection input (0 = use the odometry frame) |
| `label_transfer_radius` | 0.5 m | Radius for copying labels onto the odometry frame |
| `hybrid_enabled` / `hybrid_min_range` / `hybrid_max_range` / `hybrid_max_points` | true / 20 / 40 m / 8000 | Extra raw points far away, used only to decide |
| `vis_raw` / `vis_raw_max_range` / `vis_raw_max_points` | true / 70 m / 60000 | Past range images from the raw scan |
| `num_threads` | 4 | Parallel loops |

### Ground and height

| Key | Default | Meaning |
|---|---|---|
| `ground_band_m` | 0.25 m | Voxels below this height above ground are not scored |
| `ground_cell_m` | 2.0 m | Cell for the local ground estimate |
| `point_ground_cut` / `point_ground_cut_k` / `point_ground_cut_min_range` | 0.2 m / 0 / 10 m | Beyond 10 m, points < 0.2 m above ground stay static |
| `foot_min_hag` | 0.10 m | Near-ground voxels under a tall confirmed box stay dynamic (feet, wheels) |

### Centroid shift

| Key | Default | Meaning |
|---|---|---|
| `compare_baseline_frames` | 5 | Shift measured against t−5 |
| `noise_sigma0` / `noise_sigma_slope` / `noise_sigma_max` | 0.045 / 0.007 / 0.10 | Centroid noise σ(r) = min(max, σ0 + slope·r) — **re-tune per sensor** |
| `noise_deadzone_k` | 3.0 | Dead zone = k·σ(r) |
| `pose_err_trans_k` / `pose_err_rot_k` | 0.01 / 0.01 | Pose uncertainty added to σ |
| `w_shift` | 1.2 | Weight of the shift beyond the dead zone (in voxels) |

### Free space

| Key | Default | Meaning |
|---|---|---|
| `visibility_enabled` | true | Range-image free-space test |
| `visibility_min_age` / `visibility_max_age` / `visibility_age_step` | 6 / 14 / 4 | Past scans used: t−6, t−10, t−14 |
| `visibility_res_deg` | 0.5° | Range-image resolution |
| `visibility_min_filter` / `visibility_min_filter_rows` | 1 / −1 | Half-size of the minimum filter (3×3); rows −1 = same; sparse rings need ~ring spacing / resolution (VLP-16: 4). **0 makes precision collapse** |
| `visibility_abs_thr` / `visibility_rel_thr` | 0.3 m / 0.03 | Margin "the ray used to go farther by at least…" |
| `w_visibility` | 0.5 | Weight of the free-space fraction in the voxel score |
| `fs_enabled` | true | Free-space world map |
| `fs_res` | 0.3 m | Map cell |
| `fs_min_free` / `fs_free_ratio` | 5 / 5 | A cell is known free if seen free ≥ 5 times and ≥ 5 × occupied |
| `fs_dilate` | 1 | The 26 neighbours must be free too — **required** |
| `fs_ray_stride` / `fs_max_range` | 4 / 60 m | Rays integrated and their range |
| `fs_margin` / `fs_margin_rel` | 0.3 m / 0.03 | Free marking stops this far before the hit |

### Voxel score

| Key | Default | Meaning |
|---|---|---|
| `dynamic_score_threshold` | 0.08 | Base threshold |
| `tier1_threshold_factor` / `memory_threshold_factor` / `unconstrained_threshold_factor` | 0.55 / 0.70 / 3.0 | Inside a dynamic box / dynamic at t−1 / isolated |
| `w_history` / `frame_num_memory` | 0.3 / 5 | Penalty if the voxel was static in the last frames |
| `w_neighbor` | 0.2 | Added to the 26 neighbours of a dynamic voxel |
| `sparse_voxel_mode` | 1 | Sparse voxels are scored normally |

### Object decision

| Key | Default | Meaning |
|---|---|---|
| `cluster_propagation_threshold` | 0.45 | Fraction of dynamic voxels for a dynamic cluster |
| `min_obj_height` | 0.2 m | Minimum object height |
| `max_obj_bottom_hag` / `max_obj_top_hag` | 1.5 / 2.3 m | Lowest / highest point of the object above ground |
| `min_obj_speed` / `moving_vis_alt` | 0.1 m/s / 0.3 | Moving = speed ≥ 0.1 m/s, or free-space fraction ≥ 0.3 |
| `min_obj_vis` | 0.3 | Required mean free-space fraction |
| `fast_confirm_ratio_factor` / `fast_confirm_min_vis` | 2.0 / 0.2 | Removal in the same frame with very strong evidence |
| `unlock_ratio_factor` | 2.0 | "Strong motion" = ratio > 2 × threshold (unlocks static locks) |
| `locked_struct_frac` / `struct_min_age` | 0.7 / 30 | Locked-dynamic cluster lying on static structure: not removed, lock released |
| `locked_max_height` | 2.5 m | Locked-dynamic box taller than this is not removed |
| `vac_enable` | true | Vacated-space evidence |
| `vac_gap_frames` / `vac_min_disp` | 8 / 0.6 m | Reference points ≥ 8 frames old, after ≥ 0.6 m of displacement |
| `vac_frac` / `vac_min_pts` / `vac_min_seen` | 0.8 / 4 / 0.5 | Fraction / number / share of reference points seen through now |
| `vac_max_pts` / `vac_mid_tol` | 60 / 0.3 | Points stored per frame / tolerance of the midpoint check |
| `vac_require_tall` / `vac_max_len` | false / 0 | Shape test also for vacated space / max box length (0 = none) |

### Labelling and safety

| Key | Default | Meaning |
|---|---|---|
| `inflated_requires_evidence` | true | Inflated and history zones need the voxel's own evidence |
| `frame_max_dynamic_frac` | 0.10 | Anomalous-frame guard (typically a pose error) |

### Options (off by default)

| Key | Default | Meaning |
|---|---|---|
| `fut_delay`, `fut_output_delayed`, `fut_min_age`, `fut_age_step`, `fut_ratio`, `fut_max_range` | 0, false, 4, 3, 0.6, 22 m | Delayed decision with future scans (offline mapping) |
| `keep_voxel_evidence` | false | Keep dynamic voxels outside every box |
| `visibility_keep_frac` / `visibility_keep_min_points` | 2.0 / 2 | Same, for voxels with high free space (> 1 = off) |
| `evidence_enabled` (+ `evidence_*`) | false | World-frame evidence hysteresis |
| `point_refine_enabled` (+ `point_grow_*`, `point_bbox_margin`, `point_static_veto`, `point_knn`) | false | Point-level region growing |
| `pe_mode` (+ `pe_min_points`, `pe_max_range`) | 0 | Point labels without clusters |
| `split_enable` (+ `split_*`) | false | Motion core of fused clusters |
| `icp_enable` (+ `icp_*`) | false | ICP motion test for large clusters |
| `large_box_len` / `_ratio` / `_vis` | 1e9 / 0.7 / 0.3 | Stricter evidence for long boxes |
| `fs_raw`, `fs_occ_static_only`, `fs_decay`, `fs_flag_nb`, `fs_dda` | false, false, 1.0, false, false | Free-space map variants |
| `view_change_k` | 0 | Parallax term in σ |
| `min_obj_vis_max_range` | 1e9 | Apply `min_obj_vis` only within this range |
| `w_mahalanobis` | 0 | Mahalanobis shift term |

### Legacy and experimental keys

Read but without effect at the shipped values, kept so that old configs still load:
`w_cluster`, `static_cluster_penalty_factor`, `w_history_dynamic`, `history_factor`, `dynamic_cluster_ratio`, `points_limit`, `min_voxel_points`, `unmatched_dynamic_margin`, `min_shift_m` (the noise dead zone is always larger), `motion_threshold_scale`, `rotation_threshold_scale`, `cluster_motion_scale`, `cluster_rotation_scale`, `w_distance`, `w_velocity`, `velocity_static_threshold`, `visibility_max_incidence_deg`.

`config_dynamic_object_rejection_outdoor.json` is an old scratch file and is not read.

## `config_dynamic_cluster_extractor.json`

Section `dynamic_cluster_extractor`, read by [`DynamicClusterExtractor`](module1_voxel.md#3-clusters-and-tracks-dynamicclusterextractor).

| Key | Default | Meaning |
|---|---|---|
| `eps_voxel_factor` / `min_pts` | 1.5 / 2 | DBSCAN eps = 1.5 × voxel; core neighbours |
| `dbscan_voxel_hash` / `dbscan_cell_range` | true / 1 | Neighbour search on a voxel hash, ±1 cell |
| `eps_range_k` | 0 | *Option.* eps(r) = max(eps, k·r) along z; k ≈ 1.15 × ring spacing [rad]: VLP-16 0.04, OS1-64 0.010, HDL-64 0.008, OS2-128 0.0035 |
| `min_cluster_voxels` / `min_points_for_bbox` | 3 / 3 | Minimum cluster size |
| `cluster_max_range` / `cluster_max_height` / `cluster_ground_cell` | 70 m / 4 m / 2 m | Clustering region |
| `bbox_min_extent` / `bbox_max_extent` | 0.1 / 4.0 m | Box side window — vehicles longer than 4 m get no box |
| `bbox_min_volume` / `bbox_max_volume` | 1e-4 / 1000 m³ | Box volume window |
| `cluster_iou_threshold` | 0.1 | NMS overlap threshold |
| `frag_merge_gap` / `frag_merge_gap_k` | 0.5 m / 0.02 | Fragment merge gap: 0.5 + 0.02·r |
| `frag_max_len` / `frag_max_wid` / `frag_max_hgt` | 6 / 3 / 4 m | Maximum merged size |
| `use_motion_prediction` | true | Constant-velocity prediction of tracks |
| `use_centroid_association` | true | Hungarian assignment on centroids |
| `assoc_gate_base` / `assoc_gate_per_range` / `assoc_gate_per_missed` | 0.8 m / 0.03 / 0.3 | Association gate |
| `track_max_missed` | 5 | Coasting frames before a track is dropped |
| `track_bbox_history_size` | 5 | Past boxes kept per track |
| `min_dynamic_frames` | 2 | Hysteresis for confirmation |
| `track_evidence_decay` / `_bias` / `_on` / `_max` | 0.8 / 0.6 / 0.5 / 2.0 | Evidence accumulator; `bias` is the key knob (0.1 makes precision collapse) |
| `permanent_dynamic_frames` / `permanent_static_frames` | 8 / 20 | Locks (0 = off) |
| `permanent_unlock_frames` | 2 | Strong-motion frames that release a static lock |
| `permanent_release_no_motion_frames` | 3 | Frames without own motion that release a dynamic lock |
| `apply_lidar_imu_extrinsic` | true | See [Before you start](#before-you-start) |

Options (off): `fast_track_speed` / `fast_track_min_dynamic_frames`, `release_static_frames`, `permanent_release_max_disp`, `confirmed_hold_frames`, `use_obb` (+ `obb_*`), `inherit_gate` / `inherit_gate_per_range`, `coast_label`, `assoc_size_weight`.

Legacy: `knn_max_neighbors` (only without the voxel hash), `track_match_distance` / `track_match_iou` (only without centroid association), `peer_merge_distance_indoor` / `_outdoor` (bypassed by the fragment merge — this is also why `env_type` currently has no effect).

## `config_wall_filter.json`

Read by [`WallFilter`](module1_voxel.md#2-static-structure-wallfilter).

| Key | Default |
|---|---|
| `voxel_resolution` | 0.25 m |
| `rng_seed` | −1 (random; set ≥ 0 for reproducible runs) |
| `ransac_max_iterations` / `ransac_min_inliers` / `ransac_confidence` | 100 / 30 / 0.99 |
| `ransac_inlier_threshold` / `floor_ransac_inlier_threshold` | 0.07 m / 0.15 m |
| `wall_vertical_angle_deg` / `floor_ceiling_angle_deg` | 5.0° |
| `max_planes` | 8 |
| `wall_exclude_ground` / `wall_vertical_ransac` | true / true |
| `wall_bbox_max_aspect_ratio` | 5.0 |
| `floor_polar_enabled` | true |
| `floor_polar_r_bin` / `floor_polar_theta_deg` | 0.5 m / 10.0° |
| `floor_polar_seed_min_r` / `_seed_max_r` | 2.0 m / 15.0 m |
| `floor_polar_k_lowest` | 8 |
| `floor_polar_slope_threshold` | 0.1 |
| `floor_polar_max_ground_height` / `_max_vertical_spread` | 0.3 m / 0.30 m |
| `floor_height_threshold` | 0.5 m (only without the polar grid) |

Legacy: `floor_polar_flatness_threshold`, `floor_polar_height_margin`, `floor_min_inliers`.

## `config_wall_registry.json`

Read by `WallBBoxRegistry`.

| Key | Default |
|---|---|
| `overlap_threshold` | 0.3 |
| `enable_expiry` / `max_missed_frames` | true / 5 |
| `min_normal_dot` | 0.9 |
| `max_center_distance` / `max_duplicate_center_distance` | 5.0 m / 2.0 m |
| `max_wall_thickness` | 0.9 m |

Legacy: `merge_weight` (boxes are merged by union).

## `config_bbox_rejection.json`

Read by [`DynamicBBoxRejection`](module2_bbox.md) and, for the velocity-inflated zone, by Module 1.

| Key | Default |
|---|---|
| `inflate_margin` | 0.4 m |
| `max_bbox_frames` | 10 |
| `velocity_inflation_k` / `reverse_velocity_inflation_k` | 0.85 / 0.35 |
| `rear_inflation_k` <span class="unused">(overridden by `reverse_velocity_inflation_k`)</span> | 0.15 |
| `lateral_inflation_k` / `vertical_inflation_k` | 0.30 / 0.20 |
| `velocity_inflation_min` / `_max_speed` | 0.1 m/s / 3.0 m/s |
| `ellipse_box_cover_scale` | 1.41421356237 |
| `bbox_potentially_dynamic_min_track_age`, `bbox_static_min_track_age`, `bbox_min_static_velocity`, `bbox_rejection_obstacle_min_removed_points` <span class="unused">(unused)</span> | 5, 15, 0.1, 3 |

## `config_preprocess.json` — FIDO-relevant keys

| Key | Default | Effect |
|---|---|---|
| `random_downsample_target` | 10000 | Points of the odometry frame |
| `use_uniform_downsampling` | false | Plain uniform sampling instead of the random grid |
| `raw_points_target` | 0 | If > 0, the scan is first reduced to this many random points (also the raw scan seen by the rejection) |
| `k_correspondences` | 10 | Neighbours, now always computed (also with FIDO on) |

## `config_ros.json` — FIDO-relevant keys

| Key | Values | Effect |
|---|---|---|
| `dynamic_rejection_type` | `NONE` · `CLUSTER` · `BBOX` · `COMBINED` | Selects the module(s) in the ROS 2 node; anything but `NONE` makes the preprocessor keep the raw scan |
| `env_type` | `INDOOR` · `OUTDOOR` | Selects `peer_merge_distance_*`; no effect while the fragment merge is on |
| `bbox_topic` | `/onboard_detector/tracked_dynamic_obstacles` | External detector boxes for Module 2 |
| `imu_topic` / `points_topic` | `/imu/data` / `/velodyne_points` | Inputs |
