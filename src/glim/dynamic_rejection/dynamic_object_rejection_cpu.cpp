#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <spdlog/spdlog.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/config.hpp>

#include <glim/util/config.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif

#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

// ===========================================================================
// Helper: num_voxels() from a DynamicVoxelMapCPU
// ===========================================================================

static int nvox_of(const gtsam_points::DynamicVoxelMapCPU& vm) {
    return static_cast<int>(
        vm.gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());
}

// ===========================================================================
// DynamicObjectRejectionParamsCPU
// ===========================================================================

DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU() {
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));

    dynamic_score_threshold           = config.param<double>("dynamic_object_rejection", "dynamic_score_threshold",           2.5);
    tier1_threshold_factor            = config.param<double>("dynamic_object_rejection", "tier1_threshold_factor",            0.25);
    unconstrained_threshold_factor    = config.param<double>("dynamic_object_rejection", "unconstrained_threshold_factor",    3.0);
    memory_threshold_factor           = config.param<double>("dynamic_object_rejection", "memory_threshold_factor",           0.70);
    num_threads                       = config.param<int>   ("dynamic_object_rejection", "num_threads",                       4);
    w_shift                           = config.param<double>("dynamic_object_rejection", "w_shift",                           1.0);
    w_mahalanobis                     = config.param<double>("dynamic_object_rejection", "w_mahalanobis",                     0.8);
    w_cluster                     = config.param<double>("dynamic_object_rejection", "w_cluster",                     0.07);
    w_history                     = config.param<double>("dynamic_object_rejection", "w_history",                     0.09);
    w_neighbor                    = config.param<double>("dynamic_object_rejection", "w_neighbor",                    0.6);
    history_factor                = config.param<double>("dynamic_object_rejection", "history_factor",                0.5);
    frame_num_memory              = config.param<int>   ("dynamic_object_rejection", "frame_num_memory",              10);
    points_limit                  = config.param<double>("dynamic_object_rejection", "points_limit",                  0.25);
    cluster_propagation_threshold = config.param<double>("dynamic_object_rejection", "cluster_propagation_threshold", 0.5);

    spdlog::debug("[dynamic_rejection] params loaded");
}

DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() = default;


// ===========================================================================
// Construction
// ===========================================================================

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(
    const DynamicObjectRejectionParamsCPU&  params,
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(params), pose_kalman_filter_(pose_kalman_filter), last_pose_(Eigen::Isometry3d::Identity())
{
    covariance_estimation_ = std::make_unique<CloudCovarianceEstimation>(params_.num_threads);

    if (!pose_kalman_filter_) {
        pose_kalman_filter_ = std::make_shared<PoseKalmanFilter>();
    }

    spdlog::debug("[dynamic_rejection] DynamicObjectRejectionCPU constructed");
}


// ===========================================================================
// reject()  —  primary API
// ===========================================================================

DynamicRejectionResult DynamicObjectRejectionCPU::reject(
    const WallFilterResult&       wf_result,
    const PreprocessedFrame::Ptr& source_frame,
    std::vector<BoundingBox>& cluster_bboxes)
{
    spdlog::debug("[dynamic_rejection] reject begin: voxelmap={} wall_voxels={}/{}",
        static_cast<bool>(wf_result.voxelmap),
        wf_result.num_wall_voxels,
        wf_result.num_total_voxels);

    DynamicRejectionResult result;
    result.voxelmap = wf_result.voxelmap;

    // -----------------------------------------------------------------------
    // First frame: store as reference, return source unchanged
    // -----------------------------------------------------------------------
    if (voxelmap_history_.empty()) {
        spdlog::debug("[dynamic_rejection] first frame — storing reference voxelmap");
        voxelmap_history_.push_back(wf_result.voxelmap);
        pose_history_.push_back(pose_kalman_filter_->getPose());
        last_pose_ = pose_kalman_filter_->getPose();
        last_dynamic_frame_ = nullptr;
        result.static_frame  = source_frame;
        result.dynamic_frame = nullptr;
        return result;
    }

    Eigen::Isometry3d cur_pose     = pose_kalman_filter_->getPose();
    Eigen::Isometry3d T_delta_pose = cur_pose * last_pose_.inverse();
    last_pose_ = cur_pose;

    spdlog::debug("[dynamic_rejection] delta pose since last frame: translation=({:.4f},{:.4f},{:.4f})",
        T_delta_pose.translation().x(), T_delta_pose.translation().y(), T_delta_pose.translation().z());

    // -----------------------------------------------------------------------
    // Score voxels against previous frame, then propagate to neighbours
    // -----------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    score_voxels(*wf_result.voxelmap, *voxelmap_history_.back(), cluster_bboxes, T_delta_pose);
    auto t1 = std::chrono::steady_clock::now();
    spdlog::debug("[dynamic_rejection] score_voxels took {:.1f} ms",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    const int nvox = nvox_of(*wf_result.voxelmap);
    propagate_to_neighbors(*wf_result.voxelmap, nvox);
    propagate_to_clusters(*wf_result.voxelmap, cluster_bboxes);

    // -----------------------------------------------------------------------
    // Split voxel points into static / dynamic buckets
    // -----------------------------------------------------------------------
    std::vector<Eigen::Vector4d> static_pts,  dynamic_pts;
    std::vector<double>          static_int,  dynamic_int;
    std::vector<double>          static_tim,  dynamic_tim;

    collect_points(*wf_result.voxelmap,
                   static_pts,  static_int,  static_tim,
                   dynamic_pts, dynamic_int, dynamic_tim);

    // -----------------------------------------------------------------------
    // Update history ring — use pop_front pattern via erase-from-front
    // (deque would be O(1) but we keep vector to avoid changing the type)
    // -----------------------------------------------------------------------
    voxelmap_history_.push_back(wf_result.voxelmap);
    if (static_cast<int>(voxelmap_history_.size()) > params_.frame_num_memory) {
        voxelmap_history_.erase(voxelmap_history_.begin());
    }
    pose_history_.push_back(T_delta_pose);
    if (static_cast<int>(pose_history_.size()) > params_.frame_num_memory) {
        pose_history_.erase(pose_history_.begin());
    }

    // -----------------------------------------------------------------------
    // Build output frames
    // -----------------------------------------------------------------------
    if (static_pts.empty()) {
        spdlog::warn("[dynamic_rejection] no static points — returning source frame unchanged");
        result.static_frame = source_frame;
    } else {
        result.static_frame = build_frame(
            std::move(static_pts), std::move(static_int), std::move(static_tim),
            *source_frame);
    }

    if (!dynamic_pts.empty()) {
        auto dyn           = std::make_shared<PreprocessedFrame>();
        dyn->stamp         = source_frame->stamp;
        dyn->scan_end_time = source_frame->scan_end_time;
        dyn->points        = std::move(dynamic_pts);
        dyn->intensities   = std::move(dynamic_int);
        dyn->times         = std::move(dynamic_tim);
        dyn->k_neighbors   = 0;
        last_dynamic_frame_  = dyn;
        result.dynamic_frame = dyn;
    } else {
        last_dynamic_frame_  = nullptr;
        result.dynamic_frame = nullptr;
    }

    spdlog::debug("[dynamic_rejection] reject done: static={} dynamic={}",
        result.static_frame  ? result.static_frame->points.size()  : 0,
        result.dynamic_frame ? result.dynamic_frame->points.size() : 0);

    return result;
}


// ===========================================================================
// score_voxels()
// ===========================================================================

void DynamicObjectRejectionCPU::score_voxels(
    gtsam_points::DynamicVoxelMapCPU&       current,
    const gtsam_points::DynamicVoxelMapCPU& previous,
    const std::vector<BoundingBox>&         cluster_bboxes,
    const Eigen::Isometry3d&                T_delta_pose)
{
    const int    nvox       = nvox_of(current);
    const int    n_clusters = static_cast<int>(cluster_bboxes.size());
    const double inv_res    = 1.0 / current.voxel_resolution();
    const double points_thr = params_.points_limit * current.voxel_resolution() * 100.0;

    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();

    const int hist_depth = std::min(params_.frame_num_memory,
                                    static_cast<int>(voxelmap_history_.size())) - 1;

    // Mutex protects appends to dynamic_voxels / dynamic_voxels_neighbor_indices_.
    // Contended only when a voxel is classified dynamic (rare), so overhead is negligible.
    std::mutex dyn_mutex;

    const auto per_voxel = [&](int j) {
        auto& cur = current.lookup_voxel(j);

        if (cur.is_wall) {
            cur.is_dynamic    = false;
            cur.dynamic_score = 0.0;
            return;
        }
        if (cur.num_points < points_thr) {
            cur.is_dynamic    = false;
            cur.dynamic_score = 0.0;
            return;
        }
        cur.is_dynamic    = false;
        cur.dynamic_score = 0.0;

        // ---- Cluster bbox check ----
        bool in_any_bbox      = false;
        bool possibly_dynamic = false;

        for (int c = 0; c < n_clusters; ++c) {
            if (cluster_bboxes[c].contains(cur.mean)) {
                in_any_bbox = true;
                if (!cluster_bboxes[c].is_dynamic_bbox()) {
                    cur.dynamic_score -= params_.w_cluster * 0.5;
                } else {
                    cur.dynamic_score += params_.w_cluster;
                    possibly_dynamic = true;
                }
                break;
            }
        }
        if (!in_any_bbox) {
            cur.dynamic_score -= params_.w_cluster * 0.5;
        }

        // ---- Transform centroid into previous frame ----
        const Eigen::Vector3d p_trans  = T_delta_pose * cur.mean.head<3>();
        const Eigen::Vector4d p_trans4(p_trans.x(), p_trans.y(), p_trans.z(), 1.0);
        const auto            coord    = current.voxel_coord(p_trans4);
        int                   prev_idx = previous.lookup_voxel_index(coord);

        // Fallback: if exact hash misses (pose error ≥ 0.5 * voxel_size),
        // search the 26 neighbouring cells and pick the nearest centroid.
        // Bounded cost (≤ 26 lookups); tolerates up to one full voxel of drift.
        if (prev_idx < 0) {
            const double max_dist2 = (2.0 / inv_res) * (2.0 / inv_res); // (2 * voxel_res)^2
            double best_dist2      = max_dist2;
            for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const Eigen::Vector3i nc(coord.x() + dx, coord.y() + dy, coord.z() + dz);
                const int ni = previous.lookup_voxel_index(nc);
                if (ni < 0) continue;
                const double d2 = (p_trans - previous.lookup_voxel(ni).mean.head<3>()).squaredNorm();
                if (d2 < best_dist2) { best_dist2 = d2; prev_idx = ni; }
            }
        }

        if (prev_idx < 0) {
            // No previous match found even after 26-neighbour fallback.
            // For Tier-1 (confirmed dynamic bbox) the object likely moved fast enough
            // to leave the search window — mark it dynamic and propagate to neighbours.
            // For Tier-2/3 treat as inconclusive (static by default) to avoid false
            // positives from newly-appeared static objects inside misclassified bboxes.
            if (possibly_dynamic) {
                spdlog::debug("Voxel at ({}, {}, {}) is possibly dynamic.", cur.mean.x(), cur.mean.y(), cur.mean.z());
                cur.is_dynamic    = true;
                cur.dynamic_score = params_.dynamic_score_threshold + 1.0;
                const auto nbrs = get_neighbor_voxels(current, cur.mean);
                std::lock_guard<std::mutex> lock(dyn_mutex);
                dynamic_voxels.insert(dynamic_voxels.end(), nbrs.size(), j);
                dynamic_voxels_neighbor_indices_.insert(
                    dynamic_voxels_neighbor_indices_.end(), nbrs.begin(), nbrs.end());
            } else {
                cur.is_dynamic    = false;
                cur.dynamic_score = 0.0;
            }
            return;
        }

        const auto& prv = previous.lookup_voxel(prev_idx);


        // ---- 0. Recent-dynamic memory (threshold reduction only) ----
        // If the matched voxel in the previous frame was dynamic we lower the
        // effective detection threshold for this voxel.  We do NOT add a score
        // bonus: that would create a positive-feedback loop where a voxel stays
        // above threshold indefinitely via the memory alone, even after the
        // object has stopped moving.
        const bool was_recently_dynamic = prv.is_dynamic;

        // ---- 1. Centroid shift (normalised by voxel size) ----
        const Eigen::Vector3d delta = p_trans - prv.mean.head<3>();
        const double shift = delta.norm() * inv_res;

        // ---- 2. Mahalanobis distance ----
        double mahal = 0.0;
        {
            Eigen::Matrix3d cov = prv.cov.topLeftCorner<3, 3>();
            cov = 0.5 * (cov + cov.transpose());
            cov.diagonal().array() += 1e-6;

            Eigen::LDLT<Eigen::Matrix3d> ldlt(cov);
            if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                const Eigen::Vector3d sol = ldlt.solve(delta);
                if (sol.allFinite()) {
                    const double q = delta.dot(sol);
                    if (q >= 0.0) mahal = std::sqrt(q);
                }
            }
        }

        cur.dynamic_score += params_.w_shift * shift + params_.w_mahalanobis * mahal;

        // ---- History suppression ----
        // Count only frames where the voxel was present AND static.
        // Frames where the voxel was already dynamic do NOT count as evidence
        // of static behaviour — otherwise a moving object that has been tracked
        // for several frames would suppress its own future detection.
        if (hist_depth > 0) {
            int static_count = 0;
            Eigen::Vector4d cur_mean = p_trans4;

            for (int k = 0; k < hist_depth; ++k) {
                const int idx_hist    = static_cast<int>(voxelmap_history_.size()) - 1 - k;
                const auto& past      = voxelmap_history_[idx_hist];
                const auto& pose_prev = pose_history_[idx_hist];

                const int voxel_idx = past->lookup_voxel_index(past->voxel_coord(cur_mean));
                const Eigen::Vector3d tmp = pose_prev * cur_mean.head<3>();
                cur_mean = Eigen::Vector4d(tmp.x(), tmp.y(), tmp.z(), 1.0);

                if (voxel_idx >= 0 && !past->lookup_voxel(voxel_idx).is_dynamic)
                    ++static_count;
            }

            const double static_ratio = static_cast<double>(static_count) / hist_depth;
            cur.dynamic_score -= params_.w_history * static_ratio;
        }

        // ---- Threshold (three-tier, cluster-gated with dynamic memory) ----
        //
        //  Tier 1 — inside a CONFIRMED dynamic cluster bbox (possibly_dynamic):
        //           uses base_threshold * tier1_threshold_factor (default 0.25).
        //           The cluster tracker already confirmed motion with hysteresis,
        //           so we can accept small per-voxel shifts (slow walking ~0.25 m/s).
        //
        //  Tier 2 — was dynamic in the previous frame but cluster bbox absent:
        //           uses base_threshold * memory_threshold_factor (< 1.0).
        //           Requires real motion (e.g. 7-8 cm) but less than tier 1.
        //           No score bonus → stops cleanly when the object stops.
        //
        //  Tier 3 — isolated voxel, no cluster, no recent dynamic history:
        //           uses base_threshold * unconstrained_factor (high).
        //           Only very large shifts pass → suppresses lidar/odometry noise.
        double eff_threshold;
        if (possibly_dynamic) {
            eff_threshold = params_.dynamic_score_threshold * params_.tier1_threshold_factor;
        } else if (was_recently_dynamic) {
            eff_threshold = params_.dynamic_score_threshold * params_.memory_threshold_factor;
        } else {
            eff_threshold = params_.dynamic_score_threshold * params_.unconstrained_threshold_factor;
        }

        if (cur.dynamic_score > eff_threshold) {
            cur.is_dynamic  = true;
            const auto nbrs = get_neighbor_voxels(current, cur.mean);
            std::lock_guard<std::mutex> lock(dyn_mutex);
            dynamic_voxels.insert(dynamic_voxels.end(), nbrs.size(), j);
            dynamic_voxels_neighbor_indices_.insert(
                dynamic_voxels_neighbor_indices_.end(), nbrs.begin(), nbrs.end());
        }
    };  // end per_voxel lambda

    // Run in parallel if a parallel back-end is available.
    if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params_.num_threads) schedule(dynamic, 16)
        for (int j = 0; j < nvox; ++j) per_voxel(j);
    } else {
#ifdef GTSAM_POINTS_USE_TBB
        tbb::parallel_for(
            tbb::blocked_range<int>(0, nvox, 16),
            [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j < r.end(); ++j) per_voxel(j);
            });
#else
        for (int j = 0; j < nvox; ++j) per_voxel(j);
#endif
    }
}


// ===========================================================================
// propagate_to_neighbors()
// ===========================================================================

void DynamicObjectRejectionCPU::propagate_to_neighbors(
    gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox)
{
    for (int idx : dynamic_voxels_neighbor_indices_) {
        if (idx < 0 || idx >= nvox) continue;

        auto& v = voxelmap.lookup_voxel(idx);
        if (v.is_dynamic || v.is_wall) continue;

        v.dynamic_score += params_.w_neighbor;
        if (v.dynamic_score > params_.dynamic_score_threshold) {
            v.is_dynamic = true;
        }
    }
    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();
}


// ===========================================================================
// propagate_to_clusters()
// ===========================================================================
// If a percentage of voxels in a cluster are dynamic, mark the whole cluster.

void DynamicObjectRejectionCPU::propagate_to_clusters(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    std::vector<BoundingBox>& cluster_bboxes)
{
    if (cluster_bboxes.empty()) return;

    const int nvox      = nvox_of(voxelmap);
    const int n_clusters = static_cast<int>(cluster_bboxes.size());

    std::vector<int>  voxel_cluster(nvox, -1);
    std::vector<int>  dynamic_count(n_clusters, 0);
    std::vector<int>  total_count  (n_clusters, 0);

    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        for (int c = 0; c < n_clusters; ++c) {
            if (cluster_bboxes[c].contains(v.mean)) {
                voxel_cluster[j] = c;
                ++total_count[c];
                if (v.is_dynamic) ++dynamic_count[c];
                break;
            }
        }
    }

    // Determine which clusters flip to dynamic.
    // Only consider clusters whose bbox is CONFIRMED dynamic by the cluster
    // extractor.  Without this gate, a single odometry spike can push 30%+
    // of a static cluster's voxels over the shift threshold, causing the
    // entire cluster to be marked dynamic for that frame.
    std::vector<bool> cluster_is_dynamic(n_clusters, false);
    bool any_dynamic = false;
    for (int c = 0; c < n_clusters; ++c) {
        if (total_count[c] == 0) continue;
        const double ratio = static_cast<double>(dynamic_count[c]) / total_count[c];
        cluster_is_dynamic[c] = (ratio > params_.cluster_propagation_threshold);
        if (cluster_is_dynamic[c]) any_dynamic = true;
        spdlog::debug("[dynamic_rejection] cluster {}: {}/{} dynamic ({:.1f}%) -> {}",
                      c, dynamic_count[c], total_count[c], ratio * 100.0,
                      cluster_is_dynamic[c] ? "DYNAMIC" : "static");
    }

    if (!any_dynamic) return;

    for (int j = 0; j < nvox; ++j) {
        const int c = voxel_cluster[j];
        if (c < 0 || !cluster_is_dynamic[c]) continue;
        auto& v = voxelmap.lookup_voxel(j);
        if (!v.is_wall) {
            v.is_dynamic = true;
        }
    }
}


// ===========================================================================
// collect_points()
// ===========================================================================

void DynamicObjectRejectionCPU::collect_points(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    std::vector<Eigen::Vector4d>& static_pts,
    std::vector<double>&          static_int,
    std::vector<double>&          static_tim,
    std::vector<Eigen::Vector4d>& dynamic_pts,
    std::vector<double>&          dynamic_int,
    std::vector<double>&          dynamic_tim) const
{
    // Pre-size buckets to avoid repeated reallocations                  // OPT
    const int nvox = nvox_of(voxelmap);
    {
        size_t total_pts = 0;
        for (int j = 0; j < nvox; ++j)
            total_pts += voxelmap.lookup_voxel(j).voxel_points.size();
        static_pts.reserve(total_pts);
        static_int.reserve(total_pts);
        static_tim.reserve(total_pts);
        // Dynamic is typically small — reserve a fraction
        dynamic_pts.reserve(total_pts / 8);
        dynamic_int.reserve(total_pts / 8);
        dynamic_tim.reserve(total_pts / 8);
    }

    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);

        auto& pts  = v.is_dynamic ? dynamic_pts : static_pts;
        auto& ints = v.is_dynamic ? dynamic_int : static_int;
        auto& tims = v.is_dynamic ? dynamic_tim : static_tim;

        pts.insert(pts.end(),   v.voxel_points.begin(),      v.voxel_points.end());
        ints.insert(ints.end(), v.voxel_intensities.begin(), v.voxel_intensities.end());
        tims.insert(tims.end(), v.voxel_times.begin(),       v.voxel_times.end());
    }
}


// ===========================================================================
// build_frame()
// ===========================================================================

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::build_frame(
    std::vector<Eigen::Vector4d> points,
    std::vector<double>          intensities,
    std::vector<double>          times,
    const PreprocessedFrame&     source) const
{
    auto out           = std::make_shared<PreprocessedFrame>();
    out->stamp         = source.stamp;
    out->scan_end_time = source.scan_end_time;
    out->k_neighbors   = source.k_neighbors;
    out->points        = std::move(points);
    out->intensities   = std::move(intensities);
    out->times         = std::move(times);

    if (source.k_neighbors > 0 && !out->points.empty()) {
        out->neighbors = find_neighbors(
            out->points.data(),
            static_cast<int>(out->points.size()),
            source.k_neighbors);
    }

    return out;
}


// ===========================================================================
// find_neighbors()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::find_neighbors(
    const Eigen::Vector4d* points, int num_points, int k) const
{
    if (!points || num_points <= 0 || k <= 0) return {};

    gtsam_points::KdTree tree(points, num_points);
    std::vector<int>     neighbors(num_points * k);

    const auto per_point = [&](int i) {
        std::vector<size_t> k_idx(k, static_cast<size_t>(i));
        std::vector<double> k_sq(k);
        const size_t found =
            tree.knn_search(points[i].data(), k, k_idx.data(), k_sq.data());
        std::copy(k_idx.begin(), k_idx.begin() + found,
                  neighbors.begin() + i * k);
    };

    if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params_.num_threads) schedule(guided, 8)
        for (int i = 0; i < num_points; ++i) per_point(i);
    } else {
#ifdef GTSAM_POINTS_USE_TBB
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_points, 8),
            [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i < r.end(); ++i) per_point(i);
            });
#else
        for (int i = 0; i < num_points; ++i) per_point(i);
#endif
    }

    return neighbors;
}


// ===========================================================================
// get_neighbor_voxels()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::get_neighbor_voxels(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    const Eigen::Vector4d& mean) const
{
    std::vector<int> neighbors;
    neighbors.reserve(26);                                               // OPT: 3³-1 = 26 max neighbours
    const auto base = voxelmap.voxel_coord(mean);

    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        Eigen::Vector3i c(base.x() + dx, base.y() + dy, base.z() + dz); // OPT: avoid copy + 3 separate increments
        const int idx = voxelmap.lookup_voxel_index(c);
        if (idx >= 0) neighbors.push_back(idx);
    }

    return neighbors;
}

}  // namespace glim