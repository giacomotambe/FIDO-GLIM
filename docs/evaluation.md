# Evaluation

## Offline harness — `kitti_dynamic_eval`

Runs exactly the steps of `AsyncDynamicObjectRejection::run()` on a sequence in SemanticKITTI layout, without ROS, and writes the per-point labels on the **original scan indices**, so they can be compared with the `.label` files directly.

Build:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_DYNAMIC_REJECTION_TESTS=ON
make -j kitti_dynamic_eval
```

Run:

```bash
kitti_dynamic_eval --config <glim config dir> --seq <sequences/NN> --poses <poses/NN.txt> \
                   --out <dir> --pose_mode ros_cv [--start 0 --end N]
```

| Option | Meaning |
|---|---|
| `--pose_mode exact` | pose of the current frame from the dataset (upper bound) |
| `--pose_mode lag --lag N` | pose of frame i−N |
| `--pose_mode ros` | `glim_ros` without IMU, old behaviour: unstamped update with the pose of frame i−1 |
| `--pose_mode ros_cv` / `ros_cv2` | current `glim_ros` without IMU: stamped update with the pose of frame i−1 / i−2, constant-velocity prediction to the scan stamp |
| `--raw_max N` | random subset of the raw scan |
| `--uniform_frame N` | bypass the GLIM preprocessor: the frame is N uniform raw points beyond 1 m |
| `--diag 1` | also `rej/NNNNNN.diag` (per-point flags) and `rej/NNNNNN.boxes` (decision inputs of every box) |
| `--dump_input DIR` | write the rejection input of every frame, to run other methods on the same points |

Every flag takes a value. Poses are converted to the Velodyne frame with `T = Tr⁻¹ · P · Tr` (`calib.txt`), stamps come from `times.txt`.

Outputs: `<out>/NNNNNN.pred` (int32 n, then n × {int32 original index, uint8 dynamic}) for the odometry frame, `<out>/rej/NNNNNN.pred` for the rejection's own points, per-stage timings on stdout.

!!! warning "Use a config with `dynamic_rejection_type` set"
    The preprocessor keeps the raw scan only when `dynamic_rejection_type` in `config_ros.json` is not `"NONE"`. With the shipped `config_ros.json` (`"NONE"`) the harness silently runs without the uniform rejection frame, the far extra points and the raw range images, and writes nothing to `rej/`. Point `--config` to a copy with `"dynamic_rejection_type": "CLUSTER"`.

### Metrics — `kitti_dynamic_eval.py`

```bash
python3 test/kitti_dynamic_eval.py --seq <sequences/NN> --pred <out> [--json res.json] \
        [--radius 0.5] [--ignore 0 1] [--frames split.txt]
```

- Moving = label ids 251–259 (lower 16 bits).
- **Sampled**: metrics on the classified points. **Full scan**: every scan point takes the label of the nearest classified point within `--radius`, otherwise static.
- P / R / IoU / F1 per range band (0–10, 10–20, 20–50, 0–20, 0–50 m, all), per-frame mean IoU, mean dynamic share, frames above 10% dynamic, and the most frequent false-positive classes.

`test/run_kitti_offline.sh <seq> <tag> [config,…] [modes…]` stages a sequence, runs every config × pose mode in the container, evaluates and saves results, configs and commit hash. Its paths (NAS, container name) are those of the lab machine and must be adapted.

## Common pitfalls

These explain results that look like random labels (precision of a few percent):

1. **Labels compared by index on GLIM's output.** GLIM drops and reorders points; the static and dynamic frames are separate lists. Always evaluate on the original points (the harness does), and check that frames are not shifted.
2. **Pose not predicted.** Without IMU and without stamped updates, the rejection of scan *t* uses the pose of *t−1*: at 10 m/s that is ~1 m of ego-motion error and whole façades appear to move. Use `ros_cv`.
3. **Wrong extrinsic or pose convention.** `apply_lidar_imu_extrinsic: true` applies *your robot's* `T_lidar_imu`; on a public dataset set it to `false`. KITTI poses are in the left-camera frame and must be converted with `Tr`.
4. **Semantic instead of moving labels.** Parked cars are *static* in SemanticKITTI MOS; use the moving classes.
5. **Per-point times.** KITTI `.bin` scans have no per-point time and are already motion-compensated; do not deskew them with made-up times.

Quick diagnosis: dynamic share per frame low (1–3%) but precision ~7% → evaluation problem (1); high (10–40%) → pose, extrinsic or deskew (2, 3, 5).

## Results

Runs of 1–20 Oct 2026, labels on every point of the scan: HeLiMOS official test split (514 frames), label 0 ignored, pooled IoU; DOALS per-frame mean over the 10 annotated frames of each sequence (%); SemanticKITTI label 0 ignored, pooled. FIDO runs in this harness with dataset poses and `ros_cv`-style constant-velocity prediction, one configuration for every dataset except the clustering distance (`eps_range_k` 0.04 VLP-16, 0.010 OS1-64, 0.008 HDL-64, 0.0035 OS2-128; on the VLP-16 also `visibility_min_filter_rows: 4`).

### Moving-object IoU

| Dataset | IoU all ranges | IoU within 20 m | Precision | Recall |
|---|---|---|---|---|
| HeLiMOS KAIST05, Ouster OS2-128 | 0.53 | 0.61 | 0.84 | 0.60 |
| HeLiMOS KAIST05, Velodyne VLP-16 | 0.35 | 0.37 | 0.89 | 0.36 |
| DOALS (8 sequences, %) | 73.9 | 77.4 | — | — |
| SemanticKITTI 08 | 0.26 | 0.28 | 0.29 | 0.72 |

DOALS per scene, per-frame mean IoU (%), all ranges / within 20 m: Station 69.9 / 77.4, Shopville 71.4 / 72.7, HG 77.6 / 79.7, Niederdorf 76.7 / 79.7.

**CrossLab** — our robot (VLP-16 at 1.07 m, IMU) in a motion-capture room, 11 sequences, 6 300 scans, one person walking 0.7–6 m from the sensor; labels generated from Qualisys (points within 0.45 m of the person's vertical axis), pooled: precision 0.95, recall 0.95, IoU 0.900.

Before the lock release on static structure (`locked_struct_frac`, `struct_min_age`, `permanent_release_no_motion_frames`) FIDO reached 0.753 here: a track locked dynamic on the person was handed over to a cabinet and kept removing furniture for over 100 frames. On the other datasets the release changes IoU by −0.001 to +0.005.

### Cost per scan

FIDO's own timer, full scans, without GLIM's preprocessing, on 4 performance cores of an i7-13650HX (CPU only):

| VLP-16 | OS2-128 | DOALS hauptgebaeude_1 | DOALS station_1 |
|---|---|---|---|
| 24 ms | 52 ms | 38 ms | 52 ms |

**Online in GLIM at 10 Hz** (preprocessing, FIDO, CT odometry, local and global mapping, real-time playback):

| Sensor | FIDO mean / p99 [ms] | Scans over 100 ms | Pose latency without / with FIDO [ms] |
|---|---|---|---|
| HeLiMOS VLP-16 | 34 / 46 | 0% | 28 / 56 |
| DOALS station_1 (OS1-64) | 63 / 73 | 0% | 39 / 94 |
| HeLiMOS OS2-128 | 71 / 89 | 0% | 41 / 105 |

### Effect on GLIM's trajectory

RMSE against the reference poses, ATE after SE(3) alignment, RPE over 1 m.

- **DOALS** (8 sequences): unchanged within ±2% except Shopville, the scene with the most pedestrians close to the sensor, where RPE drops by 13–17% and ATE by up to 10%.
- **HeLiMOS Ouster** (whole KAIST05, 6.9 km): ATE 2.23 vs 2.20 m, same local drift; FIDO keeps up with real time over all 12 466 scans.
- **HeLiMOS VLP-16**: LiDAR-only odometry at car speed drifts ~9 m per 100 m either way and the median step error is the same, but with the range-dependent clustering distance FIDO doubles the rare large jumps (RPE RMSE 25 instead of 14 cm/m).
- **CrossLab** (against Qualisys): mean ATE 2.5 vs 2.4 cm, RPE 3.6 cm/m in both.
- **Marina di Pisa** (robot standing still on a street, 18 min, GPU odometry with IMU): largest excursion 5.0 vs 7.4 mm, end-to-start offset 2.6 vs 5.6 mm, per-scan jitter −10%; ATE around the mean 1.45 vs 1.37 mm.
