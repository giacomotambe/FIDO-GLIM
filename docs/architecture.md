# Architecture

FIDO runs as a **producer&ndash;consumer** system decoupled from GLIM's preprocessing thread, so scan filtering never blocks scan acquisition. The producer pushes a frame; a background thread runs the three-stage voxel pipeline and fans its results out into four lock-free queues the producer drains whenever it's ready.

![FIDO producer–consumer architecture: GLIM's preprocessing thread hands each frame to AsyncDynamicObjectRejection, a background thread that runs WallFilter, then DynamicClusterExtractor, then DynamicObjectRejectionCPU, and fans the results out into four queues consumed by GLIM odometry and diagnostics.](assets/architecture.jpeg)

## Clustering runs *before* scoring

`DynamicClusterExtractor::extract_clusters()` produces this frame's cluster boxes plus `get_dynamic_track_history()` (past boxes of already-confirmed-dynamic tracks), and both are passed into:

```cpp
DynamicObjectRejectionCPU::reject(wf, frame, cluster_bboxes, historical_bboxes)
```

so per-voxel scoring already knows which voxels sit inside corroborated dynamic evidence — this is what lets [Module 1](module1_voxel.md)'s tiered threshold work. After scoring, `update_dynamic_feedback(cluster_bboxes)` writes the outcome back into each track's hysteresis counters, closing the loop for the next frame.

!!! note "Header comment vs. actual order"
    `async_dynamic_object_rejection.hpp`'s doc comment lists the steps as WallFilter → reject → extract_clusters. The actual runtime order in `async_dynamic_object_rejection_cpu.cpp::run()` is WallFilter → **extract_clusters** → **reject** → `update_dynamic_feedback` — clustering has to happen first, since `reject()` takes `cluster_bboxes` as an input parameter.

## Ego-motion

Ego-motion between consecutive scans (ΔT, used for the motion-adaptive threshold in [Module 1](module1_voxel.md#4-dynamic-scoring-dynamicobjectrejectioncpu) and to re-express historical boxes in the current sensor frame) comes from `PoseKalmanFilter`: an error-state Kalman filter (9-dim: δp, δv, δθ) that integrates IMU measurements between SLAM pose updates and fuses them on arrival of a new SLAM pose.

## Scope of this repository

This repository builds `libglim.so` and its plugin modules (odometry, mapping, viewer, dynamic rejection) — the algorithms and the JSON schema they read. The ROS 2 node that instantiates `AsyncDynamicObjectRejection` / `DynamicBBoxRejection`, subscribes to sensor topics, and publishes the diagnostic point clouds lives in the downstream [`glim_ros2`](https://github.com/koide3/glim_ros2) workspace package that links against this library — no publisher/subscriber code exists inside this repository itself.
