# Home

## Introduction

![FIDO-GLIM](assets/logo_purple.png "FIDO-GLIM Logo"){ width="140" }

**FIDO** (Filtering and Identification of Dynamic Objects) is a dynamic-object rejection front-end built into [GLIM](https://github.com/koide3/glim), a versatile and extensible range-based 3D mapping framework. This repository *is* GLIM 1.3.0 with a fifth module tree, `src/glim/dynamic_rejection/`, added in place — everything else (`preprocess/`, `odometry/`, `mapping/`, `viewer/`) is unmodified upstream GLIM.

FIDO intercepts raw LiDAR scans before they reach GLIM's odometry front-end, splits each point cloud into a **static** subset forwarded to GLIM and a **dynamic** subset held back for diagnostics, and keeps pedestrians, vehicles and other moving agents out of the scan-matching factors and the reconstructed map.

- ***Two independent filters:*** a voxel-based, detector-free pipeline (Module 1) and a supervised bounding-box pipeline (Module 2) — selectable, or combinable, per session via `dynamic_rejection_type`.
- ***Closed feedback loop:*** cluster-level tracking (with hysteresis and a permanent-state lock) and per-voxel scoring reinforce each other frame over frame, letting Module 1 catch slow-moving objects that fall below any single-frame geometric threshold.
- ***Decoupled from SLAM:*** FIDO runs as a producer&ndash;consumer background thread (`AsyncDynamicObjectRejection`), so filtering never blocks scan acquisition, and CPU-only, so GLIM's GPU stays dedicated to scan matching.
- ***Everything else is stock GLIM:*** direct multi-scan registration error minimization on factor graphs, GPU-accelerated scan matching, interactive map correction, and support for any range sensor (spinning LiDAR, non-repetitive scan LiDAR, solid-state LiDAR, RGB-D).

!!! tip
    New here? Start with [Quickstart](quickstart.md) for the build-and-run path, or jump straight to [Architecture](architecture.md) to see how FIDO's background thread fits into GLIM's pipeline.

## The two modules

| Module | Mode | What it does |
|---|---|---|
| **[Module 1 — Voxel-based](module1_voxel.md)** | Autonomous | Geometry-only pipeline: voxelize → RANSAC wall/floor filtering → DBSCAN clustering + tracking → adaptive per-voxel scoring → label propagation. Needs no external detector. |
| **[Module 2 — Bounding-box](module2_bbox.md)** | Supervised | Consumes oriented 3D boxes from an external detector/tracker over a ROS 2 topic and strips every point inside a velocity-inflated footprint. |

## Contact

giacomotambe [:material-github:](https://github.com/giacomotambe)
Extends the work of Kenji Koide [:material-home:](https://staff.aist.go.jp/k.koide/) [:material-mail:](mailto:k.koide@aist.go.jp), National Institute of Advanced Industrial Science and Technology (AIST), Japan.
