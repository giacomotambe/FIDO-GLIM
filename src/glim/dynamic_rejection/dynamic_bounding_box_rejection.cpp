
#include <glim/dynamic_rejection/dynamic_bounding_box_rejection.hpp>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

#include <glim/util/config.hpp>
#include <glim/util/convert_to_string.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif
namespace glim {
DynamicBBoxRejection::DynamicBBoxRejection(const std::vector<BoundingBox>& bbox)
    : bboxes_(bbox) {
    Config config(GlobalConfig::get_config_path("config_bbox_rejection"));
    inflate_margin_ = config.param<double>("param_bbox_rejection", "inflate_margin",           0.0);
    v_fwd_k_        = config.param<double>("param_bbox_rejection", "velocity_inflation_k",     3.0);
    v_rear_k_       = config.param<double>("param_bbox_rejection", "rear_inflation_k",         0.45);
    v_lat_k_        = config.param<double>("param_bbox_rejection", "lateral_inflation_k",      0.60);
    v_min_          = config.param<double>("param_bbox_rejection", "velocity_inflation_min",   0.05);
    spdlog::info("DynamicBBoxRejection: inflate_margin={} v_fwd_k={} v_rear_k={} v_lat_k={} v_min={}",
        inflate_margin_, v_fwd_k_, v_rear_k_, v_lat_k_, v_min_);
}


DynamicBBoxRejection::~DynamicBBoxRejection() = default;

PreprocessedFrame::Ptr DynamicBBoxRejection::reject(const PreprocessedFrame::Ptr frame) {
    std::vector<Eigen::Vector4d> filtered_points;
    std::vector<double> filtered_intensities;
    std::vector<double> filtered_times;

    std::vector<Eigen::Vector4d> dynamic_points;
    std::vector<double> dynamic_intensities;
    std::vector<double> dynamic_times;

    for (size_t i = 0; i < frame->points.size(); ++i) {
        const Eigen::Vector4d& point = frame->points[i];
        bool is_dynamic = false;
        for (const auto& bbox : bboxes_) {
            if (bbox.contains(point) ||
                bbox.contains_inflated(point, v_fwd_k_, v_rear_k_, v_lat_k_, v_min_)) {
                is_dynamic = true;
                dynamic_points.push_back(point);
                if (!frame->intensities.empty()) {
                    dynamic_intensities.push_back(frame->intensities[i]);
                }
                dynamic_times.push_back(frame->times[i]);
                break;
            }
        }
        if (!is_dynamic) {
            filtered_points.push_back(point);
            if (!frame->intensities.empty()) {
                filtered_intensities.push_back(frame->intensities[i]);
            }
            filtered_times.push_back(frame->times[i]);

        }
    }

    PreprocessedFrame::Ptr preprocessed(new PreprocessedFrame);
    preprocessed->stamp = frame->stamp;
    preprocessed->scan_end_time = frame->scan_end_time;

    preprocessed->times = filtered_times;
    preprocessed->points = filtered_points;
    if (!frame->intensities.empty()) {
        preprocessed->intensities = filtered_intensities;
    }

    preprocessed->k_neighbors = frame->k_neighbors;
    preprocessed->neighbors = find_neighbors(filtered_points.data(), static_cast<int>(filtered_points.size()), frame->k_neighbors);
    spdlog::trace("preprocessed: {} -> {} points", frame->size(), preprocessed->size());

    if (!dynamic_points.empty()) {
        last_dynamic_frame = std::make_shared<PreprocessedFrame>();
        last_dynamic_frame->stamp = frame->stamp;
        last_dynamic_frame->scan_end_time = frame->scan_end_time;
        last_dynamic_frame->points = dynamic_points;
        last_dynamic_frame->times = dynamic_times;
        if (!frame->intensities.empty()) {
            last_dynamic_frame->intensities = dynamic_intensities;
        }
    }

    return preprocessed;
}

void DynamicBBoxRejection::insert_bounding_boxes(BoundingBox& bbox) {
    bbox.inflate(inflate_margin_);
    bboxes_.push_back(bbox);
}

std::vector<int> DynamicBBoxRejection::find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
  gtsam_points::KdTree tree(points, num_points);
  int num_threads=4;
  std::vector<int> neighbors(num_points * k);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    size_t num_found = tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + num_found, neighbors.begin() + i * k);
  };

  if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not enabled" << std::endl;
    abort();
#endif
  }

  return neighbors;
}

}  // namespace glim