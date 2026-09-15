# Getting started

## Prerequisite

1. Install FIDO-GLIM on your system following [the installation section](installation.md), or use the [Docker images](docker.md).
2. Confirm your sensor and topic configuration in `config/`:

```json
glim/config/config_ros.json
  "imu_topic": "/imu/data",
  "points_topic": "/velodyne_points",
  "bbox_topic": "/onboard_detector/tracked_dynamic_obstacles",
  "dynamic_rejection_type": "NONE",   // "NONE", "CLUSTER", "BBOX" or "COMBINED"
  "env_type": "INDOOR"                // "INDOOR" or "OUTDOOR"
```

!!! tip
    `dynamic_rejection_type` is off (`"NONE"`) by default — set it to `"CLUSTER"` (Module 1), `"BBOX"` (Module 2), or `"COMBINED"` (both) to actually enable FIDO. See [FAQ](faq.md#which-dynamic_rejection_type-should-i-use) for guidance on which to pick.

!!! warning
    On ROS 2 you need to re-run `colcon build` to apply config changes in the installed package — use `--symlink-install` to avoid this.

## Executables

FIDO-GLIM inherits GLIM's two ROS executables, unchanged: ***glim_rosnode*** and ***glim_rosbag***.

### glim_rosnode

***glim_rosnode*** launches GLIM (with FIDO's rejection stage running ahead of odometry, if enabled) as a standard ROS node subscribing to points, IMU, image and — when Module 2 is active — bounding-box topics.

```bash
ros2 run glim_ros glim_rosnode
```

### glim_rosbag

***glim_rosbag*** reads directly from a rosbag, automatically adjusting playback speed to avoid data drop while mapping in minimum time.

```bash
ros2 run glim_ros glim_rosbag <bag_name>
```

!!! note "Where the ROS node lives"
    This repository builds `libglim.so` and its plugin modules — the algorithms and the JSON config schema they read. `glim_rosnode`/`glim_rosbag` themselves, and the publishers for FIDO's diagnostic clouds, live in the downstream [`glim_ros2`](https://github.com/koide3/glim_ros2) workspace package that links against this library. See [Architecture](architecture.md) for exactly where FIDO's thread sits in that pipeline.

## Configuration files

GLIM reads parameters from JSON files in a config root directory (`glim/config` by default). It first reads `config.json`, which maps out relative paths to every submodule's parameter file, including five FIDO-specific ones:

```json
{
  "global": {
    "config_wall_filter":               "config_wall_filter.json",
    "config_bbox_rejection":            "config_bbox_rejection.json",
    "config_dynamic_object_rejection":  "config_dynamic_object_rejection.json",
    "config_wall_registry":             "config_wall_registry.json",
    "config_dynamic_cluster_extractor": "config_dynamic_cluster_extractor.json"
  }
}
```

Comments (`//`, `/* */`) are allowed — `glim::Config` strips them before parsing, which is why every shipped file documents its own fields inline.

!!! info
    See [Configuration reference](parameters.md) for every FIDO parameter, its default, and what reads it.

## What happens to a scan

Once `dynamic_rejection_type` is set, every incoming scan is split into a static frame (forwarded into GLIM's normal odometry path) and a dynamic frame (published separately for diagnostics/visualization) before it ever reaches scan matching. The full mechanics are in:

- [Architecture](architecture.md) — the producer&ndash;consumer thread and its queues
- [Module 1 — Voxel pipeline](module1_voxel.md) — the five-stage autonomous filter
- [Module 2 — Bounding box](module2_bbox.md) — the detector-driven filter

## Setup your own sensor

FIDO adds no sensor-specific requirements beyond upstream GLIM. See the upstream [Sensor setup guide](https://github.com/koide3/glim/wiki/Sensor-setup-guide).
