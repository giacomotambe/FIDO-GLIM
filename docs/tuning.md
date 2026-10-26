# Tuning guide

How to adapt Module 1 to a new sensor or scene. Effects: **P** = precision (fewer false positives), **R** = recall (fewer missed objects), **ms** = time. Every parameter is listed in the [Configuration reference](parameters.md).

## 1 · Measure first

- **With ground truth** (HeLiMOS, SemanticKITTI, DOALS): P, R and IoU = TP / (TP + FP + FN) per range band (0–10, 10–20, 20–50 m), computed on the **original scan points**, not on GLIM's reordered output. The [offline harness](evaluation.md) does exactly this.
- **Without ground truth**, use proxies: share of points removed on stable structure (walls, poles) for false positives; flicker (points changing label between frames); share of points with free-space evidence that are removed, per range band, for recall.
- **Run-to-run noise.** Parallel loops and a random RANSAC seed make identical runs differ by about ±0.7–1 IoU point. Compare on 2–3 runs, or set `num_threads: 1` and `wall_filter/rng_seed` ≥ 0. A gain below one point from a single run is not reliable.
- **Two sequences**: tune on one, check on the other.

## 2 · Recommended order

1. **Pose and extrinsic.** If they are wrong nothing else helps. Symptom: walls and façades removed, frequent "anomalous frame" guard. Check `apply_lidar_imu_extrinsic` and that the pose of the current scan is predicted ([Module 1, step 0](module1_voxel.md#0-pose-of-the-scan-posekalmanfilter)).
2. **Ground** — `ground_band_m`, `point_ground_cut`, `wall_filter/floor_polar_*`.
3. **Sensor noise** — `noise_sigma*` (see §4): it sets the dead zone of the centroid shift.
4. **Free space** — `visibility_*`, `fs_*`: ages and resolution for your sensor and object speeds.
5. **Decision thresholds** — `cluster_propagation_threshold`, `track_evidence_bias`, `min_obj_height`, `min_obj_vis`.
6. **Clustering** — `eps_voxel_factor`, `eps_range_k`, `frag_*`, `bbox_max_extent`, only if objects look split or fused.
7. **Labelling** — inflation in `config_bbox_rejection.json`, `point_ground_cut`, to move the P↔R balance.

## 3 · From symptom to parameter

| Symptom | Look at |
|---|---|
| Walls/façades removed, "everything moves" | pose prediction, `apply_lidar_imu_extrinsic`; then `pose_err_*` |
| False positives on the ground | `ground_band_m`, `point_ground_cut`, `wall_filter/floor_*` |
| False positives on vegetation and edges | `visibility_abs_thr` ↑, `fs_min_free` ↑, `min_obj_height` ↑, `track_evidence_bias` ↑ |
| Furniture or parked objects removed long after they stopped | `locked_struct_frac`, `permanent_release_no_motion_frames`, `locked_max_height` |
| Slow or far pedestrians never caught | `visibility_*_age` ↑, `noise_deadzone_k` ↓, `hybrid_*`, `vac_*` |
| Objects caught late | `min_dynamic_frames` 1, `fast_confirm_*` ↓, `track_evidence_on` ↓ |
| Objects flickering | `track_evidence_decay` ↑, `fut_delay` (offline) |
| Objects split into pieces | `frag_merge_gap` ↑; on sparse sensors `eps_range_k` |
| People fused with walls or poles | `eps_voxel_factor` ↓ |
| Cars not caught | `bbox_max_extent` (4 m is the limit; 6.5 m cost 4 IoU points on HeLiMOS) |
| Trails behind objects | `velocity_inflation_k` ↓, `track_bbox_history_size` ↓ |
| Too slow | `hybrid_max_points` ↓, `fs_ray_stride` ↑, `fs_max_range` ↓, `cluster_max_range` ↓, `visibility_res_deg` ↑ |

## 4 · A new sensor

1. **Centroid noise** (`noise_sigma0`, `noise_sigma_slope`): with the robot still in a static scene, measure the voxel-centroid shift between frames at several ranges. σ0 ≈ the value close to the sensor, slope ≈ growth per metre.
2. **Range-image resolution** (`visibility_res_deg`): about the angular spacing of the points actually used (after sampling), or slightly coarser. Too fine leaves holes; too coarse lets an object cover its own past.
3. **Vertical filter on sparse rings** (`visibility_min_filter_rows`): with 16–32 rings the image has empty rows between rings; set the vertical half-size to about ring spacing / resolution (VLP-16 at 0.5°: 4).
4. **Free-space ages**: a pedestrian (~1.3 m/s) must move at least 2–3 pixels at the range you care about. At 15 m and 0.5° a pixel is ~0.13 m, so ≥ 0.3–0.4 m, i.e. ≥ 0.3 s.
5. **Clustering on sparse rings** (`eps_range_k`): ≈ 1.15 × vertical ring spacing in radians, so the rings of one person still connect far away.
6. **Density**: `rej_input_points`, `hybrid_*`, `min_pts`.
7. **Mounting**: sensor height (ground), `cluster_max_height`, `ground_band_m`, `max_obj_bottom_hag` / `max_obj_top_hag`.
8. Only then the decision thresholds (§2, step 5).

## 5 · Method

- **One parameter at a time**, 3–4 values (×0.5, ×1, ×1.5, ×2), compared per range band.
- **Watch the video** of the extreme variants: understanding *why* something changes is worth more than +0.3 IoU.
- **Diagnose by category** before touching parameters. Missed points: no cluster · static cluster · dynamic but not confirmed · confirmed but not removed. False positives: ground in the box · inflated zone · non-existent object. The harness option `--diag 1` writes the per-box decision inputs needed for this.
- **Automatic search** once single parameters are sensible: random or Bayesian search (e.g. Optuna) over 8–12 parameters, IoU < 20 m on the tuning sequence as objective with a time constraint, checked on the other sequence.
- **Upper bounds**: replace one stage at a time with ground truth (clusters, tracking, decision) to know where work pays off. On HeLiMOS: perfect per-cluster decision → 84.6 IoU, perfect tracking → 81.2.

## 6 · Reference configurations

| Scenario | Changes from the defaults |
|---|---|
| Vehicle outdoor, Ouster 128 (HeLiMOS) | defaults |
| Offline mapping, best IoU | `fut_delay: 10`, `fut_output_delayed: true` |
| Indoor robot, low speeds | `track_evidence_on: -1` if stable structure gets removed; `ground_band_m` 0.15–0.2 |
| Sparse sensor (16–32 rings) | `visibility_res_deg` 0.7–1°, `visibility_min_filter_rows` ≈ ring spacing / resolution, `eps_range_k` (VLP-16 0.04), `min_pts` 1–2, re-tune `noise_sigma*` |
| Limited time budget | `fs_enabled: false`, `hybrid_max_points` 4000, `cluster_max_range` 40 |
