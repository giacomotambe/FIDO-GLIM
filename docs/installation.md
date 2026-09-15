# Installation

!!! tip
    Prebuilt [Docker images](docker.md) are available if you'd rather skip building from source.

## Prerequisites

| Dependency | Notes |
|---|---|
| ROS 2 Jazzy+ | Communication layer for the [FIDO-GLIM-ROS2](https://github.com/giacomotambe/FIDO-GLIM-ROS2) node |
| Eigen3, Boost (`serialization`) | Linear algebra, map serialization |
| [GTSAM](https://github.com/borglab/gtsam) ≥ 4.2 | Factor graph backend |
| [gtsam_points](https://github.com/koide3/gtsam_points) ≥ 1.2.0 | Factor graph extensions, KdTree, point-cloud & voxelmap types |
| OpenMP *or* TBB | Parallel per-voxel scoring — whichever backend `gtsam_points` was built with |
| spdlog | Structured logging (every FIDO stage logs through it) |
| [Iridescence](https://github.com/koide3/iridescence) *(optional)* | Only required if `BUILD_WITH_VIEWER=ON` |
| OpenCV *(optional)* | Only required if `BUILD_WITH_OPENCV=ON` (camera-image support) |

### Ubuntu PPAs

```bash
sudo add-apt-repository ppa:koide3/iridescence
sudo add-apt-repository ppa:koide3/gtsam
sudo apt update
sudo apt install libiridescence-dev libgtsam-no-tbb-dev
```

### gtsam_points

```bash
git clone https://github.com/koide3/gtsam_points
cd gtsam_points && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON   # OFF for a CPU-only build
make -j$(nproc)
sudo make install
```

## Build FIDO-GLIM

FIDO-GLIM replaces upstream `koide3/glim` in your ROS 2 workspace — clone it as `glim`, not alongside it.

```bash
cd ~/ros2_ws/src
git clone https://github.com/giacomotambe/FIDO-GLIM.git glim

cd ~/ros2_ws
colcon build --symlink-install --packages-select glim

source install/setup.bash
```

!!! note "Package name"
    `package.xml` declares the package as `glim` (version 1.3.0), so `--packages-select glim` is what `colcon` actually resolves.

## Build the ROS 2 node — FIDO-GLIM-ROS2

This repository only builds `libglim.so` and its plugin modules — it has no `glim_rosnode`/`glim_rosbag` executables and no publishers/subscribers of its own (see [Architecture](architecture.md#scope-of-this-repository)). Those live in a separate, companion repository, **[FIDO-GLIM-ROS2](https://github.com/giacomotambe/FIDO-GLIM-ROS2)**, which must be cloned into the same workspace and built against the `glim` package above.

```bash
cd ~/ros2_ws/src
git clone https://github.com/giacomotambe/FIDO-GLIM-ROS2.git

cd ~/ros2_ws
colcon build --symlink-install --packages-select glim glim_ros

source install/setup.bash
```

Once both are built and sourced, `ros2 run glim_ros glim_rosnode` / `glim_rosbag` are available — see [Getting started](quickstart.md#executables).

!!! note "Two repositories, one workspace"
    `FIDO-GLIM` (this repo) is the SLAM + dynamic-rejection library. `FIDO-GLIM-ROS2` is the ROS 2 wrapper around it — the node, its launch files, and the topic publishers/subscribers. Both need to sit side by side under `~/ros2_ws/src` for `colcon build` to resolve the dependency between them.

### Standalone CMake build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_WITH_CUDA=ON \
         -DBUILD_WITH_VIEWER=ON \
         -DBUILD_WITH_DYNAMIC_REJECTION_BBOX=ON
make -j$(nproc)
sudo make install
```

### Build options

| Option | Default | Effect |
|---|---|---|
| `BUILD_WITH_CUDA` | `ON` | Compiles `odometry_estimation_gpu.cpp`; falls back to `OFF` with a warning if `gtsam_points` wasn't built with CUDA. |
| `BUILD_WITH_CUDA_MULTIARCH` | `OFF` | Cross-architecture GPU build. |
| `BUILD_WITH_VIEWER` | `ON` | Builds `standard_viewer`, `interactive_viewer`, `map_editor`, `memory_monitor` against Iridescence. |
| `BUILD_WITH_OPENCV` | `ON` | Camera-image support; defines `GLIM_USE_OPENCV`. |
| `BUILD_WITH_MARCH_NATIVE` | `OFF` | Adds `-march=native` to C/C++ flags. |
| `BUILD_WITH_DYNAMIC_REJECTION_BBOX` | `ON` | Compiles [Module 2](module2_bbox.md) in; sets `GLIM_USE_DYNAMIC_REJECTION_BBOX=1`. |
| `BUILD_WITH_DYNAMIC_REJECTION_VOXEL` | `OFF` | Compiles [Module 1](module1_voxel.md)'s standalone define in; sets `GLIM_USE_DYNAMIC_REJECTION_VOXEL=1`. Mutually exclusive with the `BBOX` flag — `BBOX` wins if both are `ON`. |

!!! warning "Reading the CMake carefully"
    All `dynamic_rejection/*.cpp` sources are compiled unconditionally into `libglim` regardless of these two flags — they only toggle the `GLIM_USE_DYNAMIC_REJECTION_*` preprocessor defines consumers can branch on. A separate `BUILD_WITH_DYNAMIC_REJECTION` guard also appears further down in `CMakeLists.txt` guarding `GLIM_USE_DYNAMIC_REJECTION`, but no matching `option()` ever sets it — that branch is effectively dead. The two flags above are the ones that do something.

    Which module actually *runs* is a separate, runtime decision — see `dynamic_rejection_type` in [Configuration files](parameters.md).
