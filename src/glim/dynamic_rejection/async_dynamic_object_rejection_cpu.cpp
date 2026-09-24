#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#include <spdlog/spdlog.h>

namespace glim {

AsyncDynamicObjectRejection::AsyncDynamicObjectRejection(
    const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection,
    const std::shared_ptr<WallFilter>&                wall_filter,
    const std::shared_ptr<DynamicClusterExtractor>&   cluster_extractor)
  : dynamic_rejection_(dynamic_rejection),
    wall_filter_(wall_filter),
    cluster_extractor_(cluster_extractor)
{
    spdlog::debug("[dynamic_rejection][async] ctor");
    kill_switch     = false;
    end_of_sequence = false;
    thread          = std::thread([this] { run(); });
}

AsyncDynamicObjectRejection::~AsyncDynamicObjectRejection() {
    spdlog::debug("[dynamic_rejection][async] dtor begin");
    kill_switch = true;
    join();
    spdlog::debug("[dynamic_rejection][async] dtor end");
}

void AsyncDynamicObjectRejection::insert_frame(const glim::PreprocessedFrame::Ptr& frame) {
    spdlog::debug("[dynamic_rejection][async] insert_frame: queue_size={}",
                  input_frame_queue.size());
    input_frame_queue.push_back(frame);
}

void AsyncDynamicObjectRejection::join() {
    spdlog::debug("[dynamic_rejection][async] join begin");
    end_of_sequence = true;
    if (thread.joinable()) thread.join();
    spdlog::debug("[dynamic_rejection][async] join end");
}

int AsyncDynamicObjectRejection::workload() const {
    return static_cast<int>(input_frame_queue.size());
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_results() {
    auto results = output_frame_queue.get_all_and_clear();
    spdlog::debug("[dynamic_rejection][async] get_results: count={}", results.size());
    return results;
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_dynamic_results() {
    return dynamic_frame_queue.get_all_and_clear();
}

std::vector<WallFilterResult> AsyncDynamicObjectRejection::get_wall_results() {
    return wall_result_queue.get_all_and_clear();
}

std::vector<std::vector<BoundingBox>> AsyncDynamicObjectRejection::get_cluster_bbox_results() {
    return cluster_bbox_queue_.get_all_and_clear();
}

std::vector<std::deque<std::vector<BoundingBox>>> AsyncDynamicObjectRejection::get_cluster_history_results() {
    return cluster_history_queue_.get_all_and_clear();
}

void AsyncDynamicObjectRejection::run() {
    spdlog::debug("[dynamic_rejection][async] thread started");

    constexpr int kPrintEvery = 30;
    double sum_wall = 0.0, sum_cluster = 0.0, sum_reject = 0.0, sum_total = 0.0;
    int frame_count = 0;

    while (!kill_switch) {
        auto frames = input_frame_queue.get_all_and_clear();

        if (frames.empty()) {
            if (end_of_sequence) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        spdlog::debug("[dynamic_rejection][async] processing batch of {} frames", frames.size());
        const auto T = [](const std::chrono::steady_clock::time_point& t0) {
                        using ms = std::chrono::duration<double, std::milli>;
                        return std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();
                    };
        for (const auto& frame : frames) {
            // ------------------------------------------------------------------
            // Step 1: WallFilter — voxelize + mark wall voxels
            // ------------------------------------------------------------------
            auto t_wall = std::chrono::steady_clock::now();
            const WallFilterResult wf = wall_filter_->filter(*frame);
            const double dt_wall = T(t_wall);
            spdlog::debug("[PERF] wall_filter      {:.1f} ms  ({} vox)", dt_wall, wf.num_total_voxels);


            std::vector<glim::BoundingBox> cluster_bboxes;


            auto t_cluster = std::chrono::steady_clock::now();
            std::vector<glim::BoundingBox> historical_bboxes;
            if (cluster_extractor_ && wf.voxelmap) {
                cluster_bboxes = cluster_extractor_->extract_clusters(wf.voxelmap, frame->stamp);
                historical_bboxes = cluster_extractor_->get_dynamic_track_history();
                spdlog::debug("[dynamic_rejection][async] cluster_bboxes size={} history size={}",
                              cluster_bboxes.size(), historical_bboxes.size());
            }
            const double dt_cluster = T(t_cluster);
            spdlog::debug("[PERF] cluster_extract  {:.1f} ms  ({} bbox)", dt_cluster, cluster_bboxes.size());

            // ------------------------------------------------------------------
            // Step 2: DynamicObjectRejection — score non-wall voxels
            // ------------------------------------------------------------------
            auto t_reject = std::chrono::steady_clock::now();
            const DynamicRejectionResult dr =
                dynamic_rejection_->reject(wf, frame, cluster_bboxes, historical_bboxes);

            // Feed propagate_to_clusters() results back to the tracker so the
            // hysteresis counter (dynamic_frames) reflects confirmed dynamic detections.
            if (cluster_extractor_) {
                cluster_extractor_->update_dynamic_feedback(cluster_bboxes);
            }

            const double dt_reject = T(t_reject);
            const double dt_total  = T(t_wall);
            spdlog::debug("[PERF] reject           {:.1f} ms", dt_reject);
            spdlog::debug("[PERF] TOTAL FRAME      {:.1f} ms", dt_total);

            // ------------------------------------------------------------------
            // Media cumulativa sull'intero bag — stampa ogni kPrintEvery frame
            // ------------------------------------------------------------------
            sum_wall    += dt_wall;
            sum_cluster += dt_cluster;
            sum_reject  += dt_reject;
            sum_total   += dt_total;
            ++frame_count;
            if (frame_count % kPrintEvery == 0) {
                spdlog::debug("[VOXEL avg/{}f] wall={:.1f}ms  cluster={:.1f}ms  reject={:.1f}ms  TOTAL={:.1f}ms",
                    frame_count,
                    sum_wall    / frame_count,
                    sum_cluster / frame_count,
                    sum_reject  / frame_count,
                    sum_total   / frame_count);
            }
            // ------------------------------------------------------------------
            // Enqueue outputs
            // ------------------------------------------------------------------

            wall_result_queue.push_back(wf);
            output_frame_queue.push_back(dr.static_frame);
            cluster_bbox_queue_.push_back(cluster_bboxes);
            if (dr.dynamic_frame) {
                dynamic_frame_queue.push_back(dr.dynamic_frame);
            }

            // Always enqueue the wall result so the caller can publish wall
            // voxels even when no wall planes were found (num_wall_voxels == 0).

            
            
            spdlog::debug("[dynamic_rejection][async] frame done: "
                          "wall_voxels={}/{} static_pts={} dynamic_pts={}",
                wf.num_wall_voxels,
                wf.num_total_voxels,
                dr.static_frame  ? dr.static_frame->points.size()  : 0,
                dr.dynamic_frame ? dr.dynamic_frame->points.size() : 0);
        }
    }

    spdlog::debug("[dynamic_rejection][async] thread exited");
}

}  // namespace glim