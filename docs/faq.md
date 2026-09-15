# FAQ

### Which `dynamic_rejection_type` should I use?

`BBOX` if you already run a reliable 3D object detector/tracker and want the cheapest possible filter. `CLUSTER` ([Module 1](module1_voxel.md)) if you have no detector, or want geometry-only coverage. `COMBINED` when both are available — Module 2's boxes suppress the clear cases fast, Module 1 still scores what's left.

### What does `env_type` actually change?

Only one thing at present: which `peer_merge_distance_*` value `DynamicClusterExtractor` loads. Indoors, clusters merge only when almost coincident (0.01 m default — effectively off); outdoors, boxes within 2.8 m are fused, since a single vehicle or group of pedestrians tends to fragment into several DBSCAN clusters at longer range.

### Is FIDO GPU-accelerated?

No — every stage in `dynamic_rejection/` is CPU-bound (OpenMP or TBB, chosen by `gtsam_points::is_omp_default()` at runtime, matching whichever threading backend `gtsam_points` itself was built with). GPU cycles stay dedicated to GLIM's scan-matching factors.

### How does this relate to upstream GLIM?

This repository *is* GLIM 1.3.0 with `src/glim/dynamic_rejection/` added as a fifth module tree and five new JSON configs — everything else (`preprocess/`, `odometry/`, `mapping/`, `viewer/`) is the unmodified upstream implementation. It's meant to be built in place of `koide3/glim`, not alongside it.

### What hardware was FIDO tested on?

An **AgileX Bunker Pro** tracked UGV:

- **LiDAR** — Velodyne VLP-16, 16 beams, ±15° vertical FoV, 10 Hz, ±3 cm accuracy
- **IMU** — 6-DoF, rigidly co-mounted with the LiDAR
- **Stack** — ROS 2 Jazzy, Ubuntu 24.04

On indoor and outdoor sequences under both static and moving-robot conditions, FIDO reduces pose drift and map ghost artifacts against an unfiltered GLIM baseline — Module 1 and Module 2 combined (`COMBINED`) outperform either alone by exploiting their complementary strengths: fast detector-driven removal plus geometry-driven coverage of what the detector misses.

### Where's the ROS 2 node that publishes FIDO's diagnostic clouds?

Not in this repository — see [Architecture](architecture.md#scope-of-this-repository).

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
