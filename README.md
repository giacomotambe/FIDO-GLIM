# FIDO-GLIM

**FIDO** (Filtering and Identification of Dynamic Objects) is a dynamic object rejection framework integrated into [GLIM](https://github.com/koide3/glim), a 3D LiDAR-inertial SLAM system. FIDO intercepts raw LiDAR scans before they reach the GLIM odometry front-end, partitions each point cloud into a *static* subset (forwarded to GLIM) and a *dynamic* subset (retained for diagnostics), and thereby improves localization accuracy and map quality in environments containing moving objects.

---

## Overview

Most LiDAR SLAM pipelines assume a static world. In practice, pedestrians, vehicles, and other moving agents introduce transient measurements that corrupt scan matching and accumulate as ghost artifacts in the reconstructed map. FIDO addresses this with two complementary modules:

| Module | Mode | Description |
|--------|------|-------------|
| **Module 1 — Voxel-Based** | Autonomous | Geometry-driven pipeline: voxelization → wall/ground filtering → DBSCAN clustering → temporal tracking → adaptive per-voxel scoring |
| **Module 2 — Bounding-Box** | Supervised | Consumes 3D bounding boxes from an external detector/tracker via ROS 2 topic and removes all enclosed points |

The two modules are independent and can be used separately or combined: Module 2 acts as a fast first-pass filter for confidently detected objects, while Module 1 handles residual or detector-missed dynamic content.

---

## How It Works

### Module 1 — Voxel-Based Pipeline

Each incoming LiDAR frame passes through five sequential stages:

```
Raw scan
   │
   ▼
1. Voxelization          — hash-indexed voxel grid, per-voxel Gaussian (μ, Σ)
   │
   ▼
2. Static pre-filtering  — iterative RANSAC wall detection + persistent OBB registry
                         — polar-grid ground segmentation with RANSAC refinement
   │
   ▼
3. Cluster extraction    — DBSCAN on active voxels → bounding boxes → NMS → peer merge
                         — overlap-based tracker with hysteresis (dynamic/static counters)
   │
   ▼
4. Dynamic scoring       — per-voxel score: centroid shift + cluster context + history suppression
                         — three-tier adaptive threshold (Tier 1 < Tier 2 < Tier 3)
                         — motion-scale factor λ = 1 + α·‖Δt‖
   │
   ▼
5. Label propagation     — 26-neighbor spatial propagation
                         — cluster-level consistency enforcement
                         — tracker feedback loop (closes detection↔tracking cycle)
   │
   ├──▶ Static frame  →  GLIM odometry
   └──▶ Dynamic frame →  diagnostics / visualization
```

**Key design novelty:** a closed feedback loop between cluster-level tracking and voxel-level scoring progressively lowers the detection threshold for persistently dynamic regions, enabling detection of slow-moving objects that fall below any single-frame geometric threshold.

### Module 2 — Bounding-Box Pipeline

```
External detector (ROS 2 topic)
   │  oriented 3D bounding boxes + optional velocity
   ▼
Age filter  →  OBB containment test  →  velocity-inflated ellipsoid test
   │
   ├──▶ Static frame  →  GLIM
   └──▶ Dynamic frame →  diagnostics
```

Each box is kept active for a configurable number of frames (`max_bbox_frames`) to tolerate detector latency. When velocity is available, an asymmetric ellipsoid is inflated forward along the heading direction to cover the object's swept volume.

---

## Architecture

FIDO runs as a **producer–consumer** system decoupled from the GLIM preprocessing thread:

```
GLIM preprocessing thread
        │  insert_frame()
        ▼
┌─────────────────────────────────┐
│  AsyncDynamicObjectRejection    │  ← background thread
│  ┌──────────────────────────┐   │
│  │  WallFilter              │   │  Stage 1
│  │  DynamicClusterExtractor │   │  Stage 2
│  │  DynamicObjectRejection  │   │  Stage 3
│  └──────────────────────────┘   │
└─────────────────────────────────┘
        │
        ├──▶ static_frame_queue   →  GLIM odometry
        ├──▶ dynamic_frame_queue  →  ROS 2 publisher
        └──▶ wall_result_queue    →  ROS 2 publisher
```

The ego-motion increment ΔT between consecutive scans is provided by an **error-state Kalman filter** (`PoseKalmanFilter`) that fuses IMU predictions with SLAM pose updates.

---

## Dependencies

| Dependency | Notes |
|------------|-------|
| [GLIM](https://github.com/koide3/glim) | Base SLAM framework |
| ROS 2 (Jazzy or later) | Communication layer |
| GTSAM / gtsam_points | Factor graph, KdTree, point cloud types |
| Eigen 3 | Linear algebra |
| OpenMP or Intel TBB | Parallel per-voxel scoring (selectable at build time) |
| spdlog | Logging |

---

## Build

```bash
# Clone inside your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/<your-username>/FIDO-GLIM.git

# Build
cd ~/ros2_ws
colcon build --symlink-install --packages-select fido_glim

# Source
source install/setup.bash
```

---

## Configuration

FIDO is configured through JSON files read at startup. The main config files are:

| File | Controls |
|------|----------|
| `config_wall_filter.json` | Voxel resolution, RANSAC parameters, ground segmentation |
| `config_dynamic_cluster_extractor.json` | DBSCAN, bounding box filters, tracker thresholds |
| `config_dynamic_object_rejection.json` | Scoring weights, tier multipliers, history buffer |
| `config_bbox_rejection.json` | Velocity inflation, box margin, max box age |
| `config_ros.json` | `env_type`: `"INDOOR"` or `"OUTDOOR"` (affects peer-merge distance) |

### Key Parameters

**Voxelization**
```json
"voxel_resolution": 0.25
```

**Wall detection**
```json
"ransac_inlier_threshold": 0.07,
"ransac_confidence": 0.99,
"max_planes": 8,
"wall_vertical_angle_deg": 5.0
```

**Ground segmentation**
```json
"floor_polar_r_bin": 0.5,
"floor_polar_theta_deg": 10.0,
"floor_polar_slope_threshold": 0.1,
"floor_polar_seed_min_r": 2.0,
"floor_polar_seed_max_r": 15.0
```

**Clustering & tracking**
```json
"eps_voxel_factor": 1.5,
"min_cluster_voxels": 3,
"peer_merge_distance_indoor": 0.2,
"peer_merge_distance_outdoor": 2.0,
"min_dynamic_frames": 3,
"permanent_dynamic_frames": 10,
"permanent_static_frames": 15,
"track_max_missed": 5
```

**Scoring**
```json
"dynamic_score_threshold": 0.1,
"tier1_threshold_factor": 0.55,
"memory_threshold_factor": 0.70,
"unconstrained_threshold_factor": 3.0,
"w_shift": 1.0,
"w_mahalanobis": 0.0,
"w_cluster": 0.07,
"w_history": 0.3,
"frame_num_memory": 5,
"min_shift_m": 0.09,
"motion_threshold_scale": 0.5,
"cluster_propagation_threshold": 0.4,
"cluster_motion_scale": 5.0,
"num_threads": 4
```

**Bounding-box rejection**
```json
"max_bbox_frames": 5,
"inflate_margin": 0.0,
"velocity_inflation_k": 3.0,
"rear_inflation_k": 0.45,
"lateral_inflation_k": 0.60,
"velocity_inflation_min": 0.05,
"velocity_inflation_max_speed": 5.0
```

---

## ROS 2 Interface

### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/glim_ros/imu` | `sensor_msgs/Imu` | IMU measurements for ego-motion estimation |
| `/bounding_boxes` | `vision_msgs/Detection3DArray` (or custom) | External 3D bounding boxes for Module 2 |

### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/fido/static_points` | `sensor_msgs/PointCloud2` | Points classified as static (forwarded to GLIM) |
| `/fido/dynamic_points` | `sensor_msgs/PointCloud2` | Points classified as dynamic |
| `/fido/wall_points` | `sensor_msgs/PointCloud2` | Points belonging to detected wall planes |
| `/fido/cluster_boxes` | `visualization_msgs/MarkerArray` | Bounding boxes of detected clusters |

---

## Hardware Used

Experiments were conducted on an **AgileX Bunker Pro** tracked UGV equipped with:
- **LiDAR**: Velodyne VLP-16 (16 beams, ±15° vertical FoV, 10 Hz, ±3 cm accuracy)
- **IMU**: 6-DoF, rigidly co-mounted with the LiDAR
- **Software stack**: ROS 2 Jazzy, Ubuntu 24.04

FIDO runs entirely on CPU; GLIM uses the GPU for scan matching.

---

## Results

Evaluated on indoor and outdoor sequences under static and moving robot conditions, FIDO consistently reduces pose drift and map ghost artifacts compared to an unfiltered GLIM baseline. The combination of Module 1 and Module 2 achieves the best overall performance by exploiting complementary strengths of geometry-driven and detection-driven filtering.

---

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

---

## License

This repository extends GLIM and inherits its license. See [LICENSE](LICENSE) for details.
