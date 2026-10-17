#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <thread>
#include <memory>

#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <glim/dynamic_rejection/voxel_filtering.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/cluster_extractor.hpp>

namespace glim {

/**
 * @brief Async wrapper that runs the two-stage dynamic rejection pipeline
 *        (WallFilter → DynamicObjectRejectionCPU) on a background thread.
 *
 * Ownership model
 * ---------------
 *   - WallFilter       : shared ownership (injected at construction).
 *   - DynamicObjectRejectionCPU : shared ownership (injected at construction).
 *
 * Call sequence per frame (from the producer thread):
 *   async.insert_frame(frame);
 *
 * The background thread executes for each frame:
 *   1. wall_filter_->filter(*frame)          — voxelize + mark wall voxels
 *   2. dynamic_rejection_->reject(wf, frame) — score + split points
 *   3. cluster_extractor_->extract_clusters(wf.voxelmap) — extract dynamic clusters
 *
 * Results are retrieved from the caller's thread at any time:
 *   auto static_frames  = async.get_results();
 *   auto dynamic_frames = async.get_dynamic_results();
 */
class AsyncDynamicObjectRejection {
public:
    /**
     * @param dynamic_rejection  Scorer (DynamicObjectRejectionCPU).
     * @param wall_filter        Voxelizer + wall marker (WallFilter).
     * @param cluster_extractor  Extractor for dynamic clusters (DynamicClusterExtractor).
     */
    AsyncDynamicObjectRejection(
        const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection,
        const std::shared_ptr<WallFilter>&                wall_filter,
        const std::shared_ptr<DynamicClusterExtractor>&   cluster_extractor);

    ~AsyncDynamicObjectRejection();

    /// Push a frame into the input queue (thread-safe).
    void insert_frame(const glim::PreprocessedFrame::Ptr& frame);

    /// Signal end-of-sequence and block until the background thread finishes.
    void join();

    /// Number of frames still waiting in the input queue.
    int workload() const;

    /// Drain and return all processed static frames.
    std::vector<glim::PreprocessedFrame::Ptr> get_results();

    /// Hand each static frame to `callback` on the background thread as soon as it is ready,
    /// instead of queuing it for get_results(). Set it before the first insert_frame().
    void set_output_callback(std::function<void(const glim::PreprocessedFrame::Ptr&)> callback) {
        output_callback_ = std::move(callback);
    }

    /// Drain and return all processed dynamic-only frames.
    std::vector<glim::PreprocessedFrame::Ptr> get_dynamic_results();

    /// Drain and return WallFilterResults (one per processed frame).
    /// Each result carries the voxelmap with is_wall flags set and the list
    /// of detected wall planes — useful for publishing wall point clouds.
    std::vector<WallFilterResult> get_wall_results();

    /// Expose the most-recent voxelmap for external visualization.
    gtsam_points::DynamicVoxelMapCPU::Ptr get_last_voxelmap() const {
        return dynamic_rejection_->get_last_voxelmap();
    }

    /// Drain and return the most recent cluster bounding boxes.
    std::vector<std::vector<BoundingBox>> get_cluster_bbox_results();

    /// Drain and return history snapshots (one deque per processed frame).
    /// Each deque contains one vector<BoundingBox> per past frame (index 0 = most recent).
    std::vector<std::deque<std::vector<BoundingBox>>> get_cluster_history_results();

private:
    void run();
    void wake();   ///< Wake the rejection thread (new frame, end of sequence or shutdown).

private:
    std::atomic_bool kill_switch;
    std::atomic_bool end_of_sequence;
    std::thread      thread;
    std::mutex              wake_mutex_;   ///< Taken briefly before notifying, so no wake-up is lost.
    std::condition_variable wake_cv_;

    ConcurrentVector<glim::PreprocessedFrame::Ptr> input_frame_queue;
    ConcurrentVector<glim::PreprocessedFrame::Ptr> output_frame_queue;
    ConcurrentVector<glim::PreprocessedFrame::Ptr> dynamic_frame_queue;
    ConcurrentVector<WallFilterResult>             wall_result_queue;
    ConcurrentVector<std::vector<BoundingBox>>              cluster_bbox_queue_;
    ConcurrentVector<std::deque<std::vector<BoundingBox>>> cluster_history_queue_;

    std::shared_ptr<DynamicObjectRejectionCPU> dynamic_rejection_;
    std::shared_ptr<WallFilter>                wall_filter_;
    std::shared_ptr<DynamicClusterExtractor>   cluster_extractor_;

    std::function<void(const glim::PreprocessedFrame::Ptr&)> output_callback_;
};

}  // namespace glim