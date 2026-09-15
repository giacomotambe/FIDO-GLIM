# Docker images

`docker/ubuntu/` ships two equivalent Dockerfiles — `Dockerfile.gcc` and `Dockerfile.llvm` — on top of `nvidia/cuda:13.1.0-devel-ubuntu24.04`, targeting ROS 2 **Jazzy**. Each one:

1. Installs the ROS base + build tooling (`libfmt-dev`, `libspdlog-dev`, `libopencv-dev`, Boost, Eigen3, GLFW/GLM/PNG/JPEG for the viewer).
2. Adds the `koide3/iridescence` and `koide3/gtsam` PPAs, and installs `libiridescence-dev` / `libgtsam-no-tbb-dev`.
3. Clones and installs [`gtsam_points`](https://github.com/koide3/gtsam_points) from source with `-DBUILD_WITH_CUDA=${WITH_CUDA}`.
4. Copies this repository in and builds it twice — once headless (`BUILD_WITH_VIEWER=OFF`) and once with the viewer on — so both image variants are validated in one pass.

```bash
docker build -f docker/ubuntu/Dockerfile.gcc \
  --build-arg WITH_CUDA=ON \
  --build-arg ROS_DISTRO=jazzy \
  -t fido-glim:gcc .
```

| Build arg | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `nvidia/cuda:13.1.0-devel-ubuntu24.04` | Base image (swap for a plain `ubuntu:24.04` for a CPU-only image). |
| `WITH_CUDA` | `OFF` | Passed straight through to `gtsam_points`'s and FIDO-GLIM's own `-DBUILD_WITH_CUDA`. |
| `ROS_DISTRO` | `jazzy` | ROS 2 distribution installed inside the image. |

!!! note
    `Dockerfile.gcc` and `Dockerfile.llvm` differ only in the compiler toolchain installed — both otherwise follow the same steps and produce equivalent images.
