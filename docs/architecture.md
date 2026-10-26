# Architecture

FIDO runs as a **producer&ndash;consumer** system decoupled from GLIM's preprocessing thread, so scan filtering never blocks scan acquisition. The producer pushes a frame with `insert_frame()`, which wakes a background thread (a condition variable, no polling); the thread drains every queued frame in order, runs the voxel pipeline and hands the results back through mutex-protected queues — static frames, dynamic frames, wall results, cluster boxes — or, if one is set, through an output callback for the static frame.

![FIDO producer–consumer architecture: GLIM's preprocessing thread hands each frame to AsyncDynamicObjectRejection, a background thread that runs WallFilter, then DynamicClusterExtractor, then DynamicObjectRejectionCPU, and fans the results out into four queues consumed by GLIM odometry and diagnostics.](assets/architecture.jpeg)

## Clustering runs *before* scoring

Per frame, in order:

1. `set_frame_stamp()` — selects the pose of this scan ([Ego-motion](#ego-motion)).
2. `make_rejection_frame()` — uniform sample of the raw scan plus far extra points.
3. `WallFilter::filter()` — voxel map with ground, wall and outlier flags.
4. `DynamicClusterExtractor::extract_clusters()` — clusters and tracks, plus `get_dynamic_track_history()` (past boxes of confirmed tracks).
5. `DynamicObjectRejectionCPU::reject()` — voxel evidence, object decision, point labels.
6. `transfer_labels()` — labels copied onto the odometry frame.
7. `update_dynamic_feedback()` — the per-frame verdicts go back to the tracker.

Clustering runs before scoring because `reject()` takes the boxes as input:

```cpp
DynamicObjectRejectionCPU::reject(wf, frame, cluster_bboxes, historical_bboxes)
```

so per-voxel scoring already knows which voxels sit inside corroborated dynamic evidence — this is what lets [Module 1](module1_voxel.md)'s tiered threshold work. After scoring, `update_dynamic_feedback(cluster_bboxes)` writes the outcome back into each track's counters and evidence, closing the loop for the next frame.

!!! note "Header comment vs. actual order"
    `async_dynamic_object_rejection.hpp`'s doc comment still lists the steps as WallFilter → reject → extract_clusters. The runtime order is the one above.

## Ego-motion

Every stage reads the **absolute pose of the current scan** from `PoseKalmanFilter::getFramePose()`: it compensates ego-motion against the stored history, moves past boxes and wall boxes into the current sensor frame, and drives the free-space map.

- **With IMU** — an error-state Kalman filter (9-dim: δp, δv, δθ) integrates the IMU between SLAM pose updates and fuses each new SLAM pose.
- **Without IMU** — the filter keeps the last two stamped SLAM poses (`update(T, stamp)`) and extrapolates them at constant velocity to the scan stamp set by `set_frame_stamp()`, up to 1 s. On HeLiMOS this matches the dataset poses (IoU < 20 m 72.8 against 73.3); using the last pose without prediction drops it to 62.4.

The filter has no config keys: noise matrices and the extrapolation limit are fixed in `transformation_kalman_filter.cpp`.

## Scope of this repository

This repository builds `libglim.so` and its plugin modules (odometry, mapping, viewer, dynamic rejection) — the algorithms and the JSON schema they read. The ROS 2 node that instantiates `AsyncDynamicObjectRejection` / `DynamicBBoxRejection`, subscribes to sensor topics, and publishes the diagnostic point clouds lives in [**FIDO-GLIM-ROS2**](https://github.com/giacomotambe/FIDO-GLIM-ROS2) — a companion workspace package, also maintained as part of this project, that links against this library. It's cloned and built alongside this repo, not merged into it — see [Installation](installation.md#build-the-ros-2-node-fido-glim-ros2) — and no publisher/subscriber code exists inside this repository itself.
