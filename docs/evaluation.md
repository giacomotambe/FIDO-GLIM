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

!!! info "Version"
    The numbers below were measured on branch `fix/dynamic-rejection` at commit `30a4e05` (Oct 2026), before the last 20 commits merged in `67d6edc`, with the shipped configuration plus `apply_lidar_imu_extrinsic: false`, `rng_seed: 0` and the CT odometry. Labels are evaluated on the **full scan** (label of the nearest classified point within 0.5 m).

| Dataset | Protocol | IoU < 20 m | IoU all | P / R (all) |
|---|---|---|---|---|
| DOALS, 8 sequences | mean per frame, 10 annotated frames per sequence, GLIM odometry | 74.8 | 71.7 | 0.87 / 0.82 |
| HeLiMOS KAIST05, Ouster OS2-128 | test split (514 frames), pooled, GLIM odometry | 0.616 | 0.561 | 0.80 / 0.65 |
| HeLiMOS KAIST05, Velodyne VLP-16 | test split, pooled, GLIM odometry | 0.211 | 0.184 | 0.70 / 0.20 |
| SemanticKITTI 07, frames 0–559 | pooled, dataset poses / GLIM odometry | 0.609 / 0.627 | 0.555 / 0.562 | 0.64 / 0.83 |

With GLIM's own odometry in the loop the results match those with dataset poses, as long as the pose of the current scan is predicted.

**Comparison with M-detector** (Wu et al., *Nature Communications* 2024, geometric, no training), run on the same scans with the official `kitti1.yaml` thresholds adapted to each sensor's field of view. M-detector uses the dataset poses, FIDO its own GLIM odometry:

| Dataset | FIDO | M-detector |
|---|---|---|
| DOALS (IoU < 20 m / all, mean per frame) | 74.8 / 71.7 | 80.3 / 75.2 |
| HeLiMOS Ouster (pooled IoU) | 0.561 | 0.360 |
| HeLiMOS VLP-16 (pooled IoU) | 0.184 | 0.194 |
| SemanticKITTI 07, frames 0–559 (pooled IoU) | 0.562 | 0.498 |

M-detector decides per point at full sensor resolution and is strongest on slow, dense scenes (DOALS, pedestrians); on a moving vehicle it flags static objects that become visible (precision 0.41 on HeLiMOS Ouster against FIDO's 0.80).

**Known limits.** On 16-beam sensors recall is low and nothing is detected beyond 20 m; objects longer than `bbox_max_extent` (4 m) get no box.

**Time** (2-core test container; about half on an i7-13650HX): VLP-16 ~100 ms per frame (66 ms with the current `main` on a static-robot street sequence), Ouster OS2-128 ~290 ms; add the CT odometry when it runs in the loop.
