# FAQ

### Which `dynamic_rejection_type` should I use?

`BBOX` if you already run a reliable 3D object detector/tracker and want the cheapest possible filter. `CLUSTER` ([Module 1](module1_voxel.md)) if you have no detector, or want geometry-only coverage. `COMBINED` when both are available — Module 2's boxes suppress the clear cases fast, Module 1 still scores what's left.

### What does `env_type` actually change?

Currently nothing. It selects which `peer_merge_distance_*` value `DynamicClusterExtractor` loads, but the peer merge is bypassed by the fragment merge (`frag_merge_gap` ≥ 0), which fuses fragments by gap and object size regardless of the environment.

### Is FIDO GPU-accelerated?

No — every stage in `dynamic_rejection/` is CPU-bound (OpenMP or TBB, chosen by `gtsam_points::is_omp_default()` at runtime, matching whichever threading backend `gtsam_points` itself was built with). GPU cycles stay dedicated to GLIM's scan-matching factors.

### How does this relate to upstream GLIM?

This repository *is* GLIM 1.3.0 with `src/glim/dynamic_rejection/` added as a fifth module tree and five new JSON configs. The preprocessor has a few FIDO options (raw scan kept for the rejection, uniform sampling, `raw_points_target`, neighbours always computed); `odometry/`, `mapping/` and `viewer/` are upstream. It's meant to be built in place of `koide3/glim`, not alongside it.

### What hardware was FIDO tested on?

An **AgileX Bunker Pro** tracked UGV:

- **LiDAR** — Velodyne VLP-16, 16 beams, ±15° vertical FoV, 10 Hz, ±3 cm accuracy
- **IMU** — 6-DoF, rigidly co-mounted with the LiDAR
- **Stack** — ROS 2 Jazzy, Ubuntu 24.04

On indoor and outdoor sequences under both static and moving-robot conditions, FIDO reduces pose drift and map ghost artifacts against an unfiltered GLIM baseline — Module 1 and Module 2 combined (`COMBINED`) outperform either alone by exploiting their complementary strengths: fast detector-driven removal plus geometry-driven coverage of what the detector misses.

### Can I run FIDO without an IMU?

Yes. Use the CT odometry (`config_odometry_ct.json`) and make sure the pose updates reaching `PoseKalmanFilter` are stamped: the pose of each scan is then predicted at constant velocity. Without prediction the rejection uses a pose one scan old, and static structure appears to move. See [Module 1, step 0](module1_voxel.md#0-pose-of-the-scan-posekalmanfilter).

### How do I measure FIDO on a dataset?

With the offline harness `kitti_dynamic_eval` (build flag `BUILD_DYNAMIC_REJECTION_TESTS`), which evaluates on the original scan points. See [Evaluation](evaluation.md), including the pitfalls that make results look like random labels.

### Why does a 16-beam LiDAR give much lower recall?

The evidence is geometric: with 2° between rings a pedestrian far away is a few points on one or two rings, the range images have empty rows, and clusters break up. Use `visibility_min_filter_rows`, `eps_range_k` and the other settings in [Tuning guide](tuning.md#4-a-new-sensor); beyond ~20 m a VLP-16 currently yields almost no detections.

### Where's the ROS 2 node that publishes FIDO's diagnostic clouds?

Not in this repository — it's in [FIDO-GLIM-ROS2](https://github.com/giacomotambe/FIDO-GLIM-ROS2), a companion repository cloned and built alongside this one. See [Installation](installation.md#build-the-ros-2-node-fido-glim-ros2) and [Architecture](architecture.md#scope-of-this-repository).

## Cite

If you use FIDO-GLIM in your research, please also cite the original GLIM paper:

```bibtex
@article{koide2024glim,
  title   = {GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors},
  author  = {Koide, Kenji and others},
  journal = {Robotics and Autonomous Systems},
  year    = {2024}
}
```

This repository extends GLIM and inherits its license — see [`LICENSE`](https://github.com/giacomotambe/FIDO-GLIM/blob/main/LICENSE) in the repository root.
