/*
    FILE: cluster_extractor.cpp
    ------------------
    Improved dynamic cluster extractor.

    Changes vs v4:
      FIX-1  Classification stability: track velocity is now used as the
             primary static/dynamic discriminator.  A track whose speed stays
             below `velocity_static_threshold` for `velocity_static_frames`
             consecutive frames is declared static regardless of the raw
             IoU/distance vote.  This fixes the case where a spatially-stable
             cluster with a consistent track ID was still flipping to DYNAMIC
             because the history IoU vote oscillated due to odometry drift.

      OPT-1  History is no longer deep-copied every frame.  Instead, the
             transformed copy is built once in classify_clusters() and reused
             by both merge_with_history() and the IoU classification loop.
             Saves O(H * M) BoundingBox copies per frame.

      OPT-2  update_tracks() candidate generation is now guarded by a cheap
             size_compatible() pre-filter before the expensive iou() call.
             Full iou() is only computed for the small subset that passes.

      OPT-3  Hysteresis step uses an unordered_map<int, Track*> built once
             per frame instead of O(N_bbox * N_tracks) linear find_if searches.

      OPT-4  merge_with_history() uses pre-computed current bbox centers and
             a cheap squared-distance pre-filter before the oriented
             containment test.

      OPT-5  classify_clusters() step 4 uses a squared-distance pre-filter
             before iou() for each history bbox.
*/

#include <glim/dynamic_rejection/cluster_extractor.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <gtsam_points/ann/kdtree.hpp>
#include <glim/util/config.hpp>

namespace glim {

// ===========================================================================
// DynamicClusterExtractorParams
// ===========================================================================

DynamicClusterExtractorParams::DynamicClusterExtractorParams() {
    spdlog::debug("[cluster_extractor] loading config");

    Config config(GlobalConfig::get_config_path("config_dynamic_cluster_extractor"));

    eps_voxel_factor    = config.param<double>("dynamic_cluster_extractor", "eps_voxel_factor",    2.0);
    min_pts             = config.param<int>   ("dynamic_cluster_extractor", "min_pts",             1);
    knn_max_neighbors   = config.param<int>   ("dynamic_cluster_extractor", "knn_max_neighbors",   64);
    min_cluster_voxels  = config.param<int>   ("dynamic_cluster_extractor", "min_cluster_voxels",  2);
    min_points_for_bbox = config.param<int>   ("dynamic_cluster_extractor", "min_points_for_bbox", 20);

    bbox_min_extent = config.param<double>("dynamic_cluster_extractor", "bbox_min_extent", 0.0);
    bbox_max_extent = config.param<double>("dynamic_cluster_extractor", "bbox_max_extent", 1e9);
    bbox_min_volume = config.param<double>("dynamic_cluster_extractor", "bbox_min_volume", 0.0);
    bbox_max_volume = config.param<double>("dynamic_cluster_extractor", "bbox_max_volume", 1e9);

    cluster_distance_threshold = config.param<double>("dynamic_cluster_extractor", "cluster_distance_threshold", 0.5);
    cluster_iou_threshold      = config.param<double>("dynamic_cluster_extractor", "cluster_iou_threshold",      0.5);
    history_size               = config.param<int>   ("dynamic_cluster_extractor", "history_size",               5);
    min_static_history_matches = config.param<int>   ("dynamic_cluster_extractor", "min_static_history_matches", 3);

    containment_margin   = config.param<double>("dynamic_cluster_extractor", "containment_margin",   0.1);
    merge_volume_ratio   = config.param<double>("dynamic_cluster_extractor", "merge_volume_ratio",   1.5);
    peer_merge_distance  = config.param<double>("dynamic_cluster_extractor", "peer_merge_distance",  0.0);

    // --- Tracking params ---
    track_match_distance  = config.param<double>("dynamic_cluster_extractor", "track_match_distance",  1.5);
    track_match_iou       = config.param<double>("dynamic_cluster_extractor", "track_match_iou",       0.3);
    track_max_missed      = config.param<int>   ("dynamic_cluster_extractor", "track_max_missed",      3);
    hysteresis_dynamic_n  = config.param<int>   ("dynamic_cluster_extractor", "hysteresis_dynamic_n",  2);
    hysteresis_static_m   = config.param<int>   ("dynamic_cluster_extractor", "hysteresis_static_m",   3);

    // FIX-1: velocity-based static discrimination params.
    velocity_static_threshold = config.param<double>("dynamic_cluster_extractor", "velocity_static_threshold", 0.05);
    velocity_static_frames    = config.param<int>   ("dynamic_cluster_extractor", "velocity_static_frames",    3);

    // EMA-based classification params.
    ema_alpha                  = config.param<double>("dynamic_cluster_extractor", "ema_alpha",                  0.3);
    dynamic_score_threshold    = config.param<double>("dynamic_cluster_extractor", "dynamic_score_threshold",    0.6);
    velocity_dynamic_threshold = config.param<double>("dynamic_cluster_extractor", "velocity_dynamic_threshold", 0.2);
    min_track_age              = config.param<int>   ("dynamic_cluster_extractor", "min_track_age",              3);
    static_history_ratio       = config.param<double>("dynamic_cluster_extractor", "static_history_ratio",       0.6);

    // Ego-motion robustness params.
    velocity_beta               = config.param<double>("dynamic_cluster_extractor", "velocity_beta",               0.8);
    motion_scale_factor         = config.param<double>("dynamic_cluster_extractor", "motion_scale_factor",         2.0);
    track_match_distance_strict = config.param<double>("dynamic_cluster_extractor", "track_match_distance_strict", 0.5);

    spdlog::debug("[cluster_extractor] eps_factor={:.2f} min_pts={} knn_max={} "
                  "min_cluster_voxels={} min_points_bbox={} "
                  "bbox_extent=[{:.2f},{:.2f}] bbox_volume=[{:.3f},{:.3f}] "
                  "cluster_distance_threshold={:.2f} cluster_iou_threshold={:.2f} "
                  "history_size={} static_history_ratio={:.2f} "
                  "containment_margin={:.2f} merge_volume_ratio={:.2f} "
                  "track_match_distance={:.2f} track_match_iou={:.2f} "
                  "track_max_missed={} "
                  "velocity_static_threshold={:.3f} velocity_static_frames={} "
                  "velocity_dynamic_threshold={:.3f} "
                  "ema_alpha={:.2f} dynamic_score_threshold={:.2f} min_track_age={}",
                  eps_voxel_factor, min_pts, knn_max_neighbors,
                  min_cluster_voxels, min_points_for_bbox,
                  bbox_min_extent, bbox_max_extent,
                  bbox_min_volume, bbox_max_volume,
                  cluster_distance_threshold, cluster_iou_threshold,
                  history_size, static_history_ratio,
                  containment_margin, merge_volume_ratio,
                  track_match_distance, track_match_iou,
                  track_max_missed,
                  velocity_static_threshold, velocity_static_frames,
                  velocity_dynamic_threshold,
                  ema_alpha, dynamic_score_threshold, min_track_age);
}

DynamicClusterExtractorParams::~DynamicClusterExtractorParams() = default;

// ===========================================================================
// Constructors
// ===========================================================================

DynamicClusterExtractor::DynamicClusterExtractor(
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(), pose_kalman_filter_(pose_kalman_filter) {}

DynamicClusterExtractor::DynamicClusterExtractor(
    const DynamicClusterExtractorParams& params)
    : params_(params) {}

// ===========================================================================
// NMS utility
// ===========================================================================

static std::vector<BoundingBox> nms3D(
    const std::vector<BoundingBox>& boxes,
    double iou_threshold)
{
    const int N = static_cast<int>(boxes.size());
    if (N == 0) return {};

    struct Candidate { int idx; double score; };
    std::vector<Candidate> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& s = boxes[i].get_size();
        candidates.push_back({i, s.x() * s.y() * s.z()});
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b){ return a.score > b.score; });

    std::vector<bool> suppressed(N, false);
    std::vector<BoundingBox> result;
    result.reserve(N);
    for (int i = 0; i < N; ++i) {
        const int ii = candidates[i].idx;
        if (suppressed[ii]) continue;
        result.push_back(boxes[ii]);
        for (int j = i + 1; j < N; ++j) {
            const int jj = candidates[j].idx;
            if (!suppressed[jj] && boxes[ii].overlap(boxes[jj]) > iou_threshold)
                suppressed[jj] = true;
        }
    }
    return result;
}

// ===========================================================================
// extract_clusters()
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::extract_clusters(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap)
{
    const auto cluster_map = cluster_voxels(voxelmap);

    int num_clusters = 0;
    for (int id : cluster_map)
        if (id >= num_clusters) num_clusters = id + 1;

    spdlog::debug("[cluster_extractor] found {} clusters", num_clusters);

    const auto clusters = build_point_clusters(voxelmap, cluster_map, num_clusters);
    auto bboxes = compute_bounding_boxes(clusters);
    bboxes = nms3D(bboxes, params_.cluster_iou_threshold);

    spdlog::debug("[cluster_extractor] {} bboxes after NMS", bboxes.size());

    if (pose_kalman_filter_) {
        Eigen::Isometry3d cur_pose = pose_kalman_filter_->getPose();
        Eigen::Isometry3d T_delta  = cur_pose * last_pose_.inverse();
        last_pose_ = cur_pose;
        classify_clusters(T_delta, bboxes);
    }

    return bboxes;
}

// ===========================================================================
// cluster_voxels()  —  DBSCAN on non-wall voxel centroids
// ===========================================================================

DynamicClusterExtractor::ClusterMap
DynamicClusterExtractor::cluster_voxels(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const
{
    const int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());

    ClusterMap cluster_map(nvox, -1);

    std::vector<int>             active_ids;
    std::vector<Eigen::Vector4d> active_cents;
    active_ids.reserve(nvox);
    active_cents.reserve(nvox);

    for (int i = 0; i < nvox; ++i) {
        const auto& v = voxelmap->lookup_voxel(i);
        if (!v.is_wall) {
            active_ids.push_back(i);
            active_cents.push_back(v.mean);
        }
    }

    const int n_active = static_cast<int>(active_ids.size());
    spdlog::debug("[DBSCAN] nvox={} active(non-wall)={}", nvox, n_active);
    if (n_active == 0) return cluster_map;

    const double voxel_res = voxelmap->voxel_resolution();
    const double eps       = params_.eps_voxel_factor * voxel_res;
    const double eps2      = eps * eps;
    const int    min_pts   = params_.min_pts;
    const int    knn_k     = std::min(n_active, params_.knn_max_neighbors);

    spdlog::debug("[DBSCAN] voxel_res={:.3f} eps={:.3f} min_pts={} knn_k={}",
                  voxel_res, eps, min_pts, knn_k);

    gtsam_points::KdTree tree(active_cents.data(), n_active);

    auto range_query = [&](int local_idx) -> std::vector<int> {
        std::vector<size_t> k_idx(knn_k);
        std::vector<double> k_sq (knn_k);
        const size_t found = tree.knn_search(
            active_cents[local_idx].data(), knn_k, k_idx.data(), k_sq.data());

        std::vector<int> neighbors;
        neighbors.reserve(found);
        for (size_t i = 0; i < found; ++i)
            if (k_sq[i] <= eps2 && static_cast<int>(k_idx[i]) != local_idx)
                neighbors.push_back(static_cast<int>(k_idx[i]));
        return neighbors;
    };

    constexpr int UNVISITED = -2;
    constexpr int NOISE_LBL = -1;

    std::vector<int>  label(n_active, UNVISITED);
    std::vector<bool> in_seed(n_active, false);
    int cluster_id = 0;

    for (int i = 0; i < n_active; ++i) {
        if (label[i] != UNVISITED) continue;

        const auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts) {
            label[i] = NOISE_LBL;
            continue;
        }

        label[i] = cluster_id;
        std::vector<int> seed_set = neighbors;
        for (int nb : neighbors) in_seed[nb] = true;

        for (int s = 0; s < static_cast<int>(seed_set.size()); ++s) {
            const int q = seed_set[s];
            if (label[q] == NOISE_LBL) { label[q] = cluster_id; continue; }
            if (label[q] != UNVISITED)  continue;

            label[q] = cluster_id;
            const auto q_nbrs = range_query(q);
            if (static_cast<int>(q_nbrs.size()) >= min_pts) {
                for (int nb : q_nbrs) {
                    if ((label[nb] == UNVISITED || label[nb] == NOISE_LBL) && !in_seed[nb]) {
                        seed_set.push_back(nb);
                        in_seed[nb] = true;
                    }
                }
            }
        }

        for (int nb : seed_set) in_seed[nb] = false;
        in_seed[i] = false;
        ++cluster_id;
    }

    spdlog::debug("[DBSCAN] raw_clusters={}", cluster_id);

    std::vector<int> cluster_size(cluster_id, 0);
    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0) ++cluster_size[label[i]];

    std::vector<int> remap(cluster_id, -1);
    int valid_id = 0;
    for (int c = 0; c < cluster_id; ++c)
        if (cluster_size[c] >= params_.min_cluster_voxels)
            remap[c] = valid_id++;

    spdlog::debug("[DBSCAN] {} clusters after size filter (min_voxels={})",
                  valid_id, params_.min_cluster_voxels);

    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0 && remap[label[i]] >= 0)
            cluster_map[active_ids[i]] = remap[label[i]];

    return cluster_map;
}

// ===========================================================================
// build_point_clusters()
// ===========================================================================

std::vector<std::vector<Eigen::Vector4d>>
DynamicClusterExtractor::build_point_clusters(
    VoxelMapPtr       voxelmap,
    const ClusterMap& cluster_map,
    int               num_clusters) const
{
    std::vector<std::vector<Eigen::Vector4d>> point_clusters(num_clusters);

    const int nvox = static_cast<int>(cluster_map.size());
    for (int i = 0; i < nvox; ++i) {
        const int cid = cluster_map[i];
        if (cid < 0) continue;
        const auto& voxel = voxelmap->lookup_voxel(i);
        point_clusters[cid].insert(
            point_clusters[cid].end(),
            voxel.voxel_points.begin(),
            voxel.voxel_points.end());
    }

    return point_clusters;
}

// ===========================================================================
// compute_bounding_boxes()
// ===========================================================================

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters) const
{
    return compute_bounding_boxes(clusters, params_.min_points_for_bbox);
}

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters,
    int min_points) const
{
    std::vector<BoundingBox> boxes;
    boxes.reserve(clusters.size());

    int rejected_pts  = 0;
    int rejected_geom = 0;

    for (const auto& cluster : clusters) {
        if (static_cast<int>(cluster.size()) < min_points) {
            ++rejected_pts;
            continue;
        }

        BoundingBox bbox(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(),
                         Eigen::Matrix3d::Identity());
        if (!createOBB(cluster, bbox)) {
            ++rejected_geom;
            continue;
        }
        boxes.push_back(bbox);
    }

    spdlog::debug("[cluster_extractor] {} bboxes kept (rejected: {} pts, {} geom, min_pts={})",
                  boxes.size(), rejected_pts, rejected_geom, min_points);
    return boxes;
}

// ===========================================================================
// createOBB()  —  AABB (axis-aligned, single pass over points)
// ===========================================================================

bool DynamicClusterExtractor::createOBB(
    const std::vector<Eigen::Vector4d>& cluster,
    BoundingBox& out_bbox) const
{
    Eigen::Vector3d pt_min( std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max());
    Eigen::Vector3d pt_max(-std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max());

    for (const auto& p : cluster) {
        pt_min = pt_min.cwiseMin(p.head<3>());
        pt_max = pt_max.cwiseMax(p.head<3>());
    }

    const Eigen::Vector3d size   = pt_max - pt_min;
    const Eigen::Vector3d center = 0.5 * (pt_min + pt_max);

    const double min_dim = size.minCoeff();
    const double max_dim = size.maxCoeff();
    const double volume  = size.x() * size.y() * size.z();

    if (params_.bbox_min_extent > 0.0 && min_dim < params_.bbox_min_extent) return false;
    if (params_.bbox_max_extent < 1e8 && max_dim > params_.bbox_max_extent) return false;
    if (params_.bbox_min_volume > 0.0 && volume < params_.bbox_min_volume)  return false;
    if (params_.bbox_max_volume < 1e8 && volume > params_.bbox_max_volume)  return false;

    out_bbox = BoundingBox(size, center, Eigen::Matrix3d::Identity());
    return true;
}

// ===========================================================================
// Anonymous utilities
// ===========================================================================

namespace {

// OPT-4: accepts pre-computed query center to avoid repeated get_center() calls.
bool contains_center(
    const BoundingBox& container,
    const Eigen::Vector3d& query_center,
    double margin)
{
    const Eigen::Vector3d delta     = query_center - container.get_center();
    const Eigen::Vector3d local     = container.get_rotation().transpose() * delta;
    const Eigen::Vector3d half_size = 0.5 * container.get_size();
    const Eigen::Vector3d limit     = half_size + Eigen::Vector3d::Constant(margin);

    return (std::abs(local.x()) <= limit.x() &&
            std::abs(local.y()) <= limit.y() &&
            std::abs(local.z()) <= limit.z());
}

// OPT-2: cheap volume-ratio pre-filter used before iou() in update_tracks().
// Two bboxes whose volumes differ by more than ratio_limit are unlikely to
// represent the same physical object.
bool size_compatible(const BoundingBox& a, const BoundingBox& b,
                     double ratio_limit = 3.0)
{
    const Eigen::Vector3d& sa = a.get_size();
    const Eigen::Vector3d& sb = b.get_size();
    const double va = sa.x() * sa.y() * sa.z();
    const double vb = sb.x() * sb.y() * sb.z();
    if (va <= 0.0 || vb <= 0.0) return false;
    const double r = va > vb ? va / vb : vb / va;
    return r <= ratio_limit;
}

} // anonymous namespace

// Compute the union OBB of two bboxes, keeping a's local frame.
static BoundingBox compute_union_bbox(const BoundingBox& a, const BoundingBox& b)
{
    const Eigen::Matrix3d& R   = a.get_rotation();
    const Eigen::Vector3d& c_a = a.get_center();
    const Eigen::Vector3d  he_a = a.get_size() * 0.5;

    const Eigen::Vector3d c_b_local  = R.transpose() * (b.get_center() - c_a);
    const Eigen::Matrix3d R_b_local  = R.transpose() * b.get_rotation();
    const Eigen::Vector3d he_b       = b.get_size() * 0.5;

    Eigen::Vector3d lo = -he_a;
    Eigen::Vector3d hi =  he_a;

    for (int sx : {-1, 1})
    for (int sy : {-1, 1})
    for (int sz : {-1, 1}) {
        const Eigen::Vector3d v = c_b_local +
            R_b_local * Eigen::Vector3d(sx * he_b.x(), sy * he_b.y(), sz * he_b.z());
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
    }

    return BoundingBox(hi - lo, c_a + R * (0.5 * (lo + hi)), R);
}

// ===========================================================================
// merge_with_history()
//
// OPT-4: uses pre-computed current bbox centers and a cheap squared-distance
// pre-filter (bbox diagonal) before the more expensive oriented containment
// test.
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::merge_with_history(
    const std::vector<BoundingBox>& current_bboxes,
    const std::deque<std::vector<BoundingBox>>& history_transformed,
    const std::unordered_set<int>& protected_track_ids) const
{
    const int N = static_cast<int>(current_bboxes.size());
    if (N == 0) return {};

    // --- Phase 0: peer merging — fonde bbox correnti vicine in bbox più grandi ---
    // Ordina per volume decrescente: le bbox più grandi assorbono quelle piccole.
    std::vector<BoundingBox> working(current_bboxes);
    if (params_.peer_merge_distance > 0.0) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return working[a].get_size().prod() > working[b].get_size().prod();
        });

        std::vector<bool>        peer_absorbed(N, false);
        std::vector<Eigen::Vector3d> centers(N);
        for (int i = 0; i < N; ++i) centers[i] = working[i].get_center();

        const double d2_max = params_.peer_merge_distance * params_.peer_merge_distance;

        for (int ii = 0; ii < N; ++ii) {
            const int i = order[ii];
            if (peer_absorbed[i]) continue;
            for (int jj = ii + 1; jj < N; ++jj) {
                const int j = order[jj];
                if (peer_absorbed[j]) continue;
                if ((centers[i] - centers[j]).squaredNorm() <= d2_max) {
                    working[i]  = compute_union_bbox(working[i], working[j]);
                    centers[i]  = working[i].get_center();
                    peer_absorbed[j] = true;
                    spdlog::debug("[merge_with_history] peer-merged bbox {} into {}", j, i);
                }
            }
        }

        // Rimuovi le bbox assorbite
        std::vector<BoundingBox> tmp;
        tmp.reserve(N);
        for (int i = 0; i < N; ++i)
            if (!peer_absorbed[i]) tmp.push_back(working[i]);
        working = std::move(tmp);

        spdlog::debug("[merge_with_history] peer merge: {} -> {} bboxes",
                      N, working.size());
    }

    const int M = static_cast<int>(working.size());

    std::vector<bool>        absorbed(M, false);
    std::vector<BoundingBox> merged_in;

    const double margin       = params_.containment_margin;
    const double volume_ratio = params_.merge_volume_ratio;

    // OPT-4: pre-compute current centers once.
    std::vector<Eigen::Vector3d> curr_centers(M);
    for (int i = 0; i < M; ++i)
        curr_centers[i] = working[i].get_center();

    for (const auto& frame_bboxes : history_transformed) {
        for (const auto& hist_bbox : frame_bboxes) {

            const Eigen::Vector3d& hs        = hist_bbox.get_size();
            const double           hist_vol  = hs.x() * hs.y() * hs.z();
            const Eigen::Vector3d& hc        = hist_bbox.get_center();

            // Cheap reject: use bbox half-diagonal + margin as distance threshold.
            const double half_diag2 = (0.5 * hs + Eigen::Vector3d::Constant(margin))
                                          .squaredNorm();

            std::vector<int> newly_absorbed_ids;

            for (int i = 0; i < M; ++i) {
                if (absorbed[i]) continue;

                // Never absorb bboxes that belong to established tracks (age > 1).
                // Replacing a well-tracked cluster with a phantom at its old
                // history position would give the downstream voxel rejector a
                // stale, static bbox instead of the actual moved cluster.
                if (protected_track_ids.count(working[i].get_track_id()) > 0)
                    continue;

                const Eigen::Vector3d& cs       = working[i].get_size();
                const double           curr_vol = cs.x() * cs.y() * cs.z();

                if (hist_vol < volume_ratio * curr_vol) continue;

                // OPT-4: squared-distance pre-filter.
                if ((curr_centers[i] - hc).squaredNorm() > half_diag2) continue;

                if (contains_center(hist_bbox, curr_centers[i], margin))
                    newly_absorbed_ids.push_back(i);
            }

            if (!newly_absorbed_ids.empty()) {
                for (int idx : newly_absorbed_ids) absorbed[idx] = true;

                // Goal-6: do NOT force static here — let the EMA tracking logic decide.
                // The replacement inherits the last stable classification from history.
                BoundingBox replacement = hist_bbox;
                replacement.set_track_id(-1);
                merged_in.push_back(replacement);

                spdlog::debug("[merge_with_history] hist bbox at ({:.2f},{:.2f},{:.2f}) "
                              "absorbed {} current bbox(es)",
                              hc.x(), hc.y(), hc.z(),
                              newly_absorbed_ids.size());
            }
        }
    }

    std::vector<BoundingBox> result;
    result.reserve(M + static_cast<int>(merged_in.size()));
    for (int i = 0; i < M; ++i)
        if (!absorbed[i]) result.push_back(working[i]);
    result.insert(result.end(), merged_in.begin(), merged_in.end());

    const int n_absorbed = static_cast<int>(
        std::count(absorbed.begin(), absorbed.end(), true));
    spdlog::debug("[merge_with_history] {} in -> {} out ({} absorbed, {} merged in)",
                  N, result.size(), n_absorbed, merged_in.size());

    return result;
}

// ===========================================================================
// update_tracks()
//
// OPT-2: size_compatible() pre-filter before iou() avoids the costly OBB
// intersection computation for clearly incompatible bbox pairs.
// Uses squaredNorm() for distance gating (avoids sqrt until needed for sort).
// ===========================================================================

void DynamicClusterExtractor::update_tracks(
    std::vector<BoundingBox>& bboxes,
    const Eigen::Isometry3d&  T_to_current,
    double                    dist_scale)
{
    const int N_tracks = static_cast<int>(tracks_.size());
    const int N_bboxes = static_cast<int>(bboxes.size());

    // 1. Transform track state into current sensor frame.
    for (auto& t : tracks_) {
        t.center   = T_to_current * t.center;
        t.velocity = T_to_current.linear() * t.velocity;
    }

    // 2/3. Predict and build candidate pairs.
    // Fix-1: accept if dist < D_strict (no overlap needed)
    //        OR   dist < D_loose AND overlap > overlap_low
    struct Pair { int t_idx, b_idx; double dist; };
    std::vector<Pair> pairs;
    pairs.reserve(std::min(static_cast<size_t>(N_tracks) * 4u,
                           static_cast<size_t>(N_tracks) * static_cast<size_t>(N_bboxes)));

    const double D_loose  = params_.track_match_distance * dist_scale;
    const double D_strict = params_.track_match_distance_strict * dist_scale;
    const double D_loose2  = D_loose  * D_loose;
    const double D_strict2 = D_strict * D_strict;
    const double overlap_low = params_.track_match_iou * 0.4;

    for (int t = 0; t < N_tracks; ++t) {
        const Eigen::Vector3d predicted = tracks_[t].center + tracks_[t].velocity;
        for (int b = 0; b < N_bboxes; ++b) {
            const double d2 = (predicted - bboxes[b].get_center()).squaredNorm();
            if (d2 >= D_loose2) continue;

            // OPT-2: cheap size pre-filter.
            if (!size_compatible(tracks_[t].last_bbox, bboxes[b])) continue;

            if (d2 < D_strict2) {
                pairs.push_back({t, b, std::sqrt(d2)});
                continue;
            }

            const double overlap_val = tracks_[t].last_bbox.overlap(bboxes[b]);
            if (overlap_val >= overlap_low)
                pairs.push_back({t, b, std::sqrt(d2)});
        }
    }

    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b){ return a.dist < b.dist; });

    // 4/5. Greedy assignment.
    std::vector<bool> track_matched(N_tracks, false);
    std::vector<bool> bbox_matched (N_bboxes, false);

    for (const auto& p : pairs) {
        if (track_matched[p.t_idx] || bbox_matched[p.b_idx]) continue;

        track_matched[p.t_idx] = true;
        bbox_matched [p.b_idx] = true;

        Track& t = tracks_[p.t_idx];
        const Eigen::Vector3d new_center = bboxes[p.b_idx].get_center();

        // Fix-3: low-pass filter on velocity to prevent noise-induced false dynamics.
        const Eigen::Vector3d raw_vel = new_center - t.center;
        t.velocity      = params_.velocity_beta * t.velocity
                        + (1.0 - params_.velocity_beta) * raw_vel;
        t.center        = new_center;
        t.last_bbox     = bboxes[p.b_idx];
        t.missed_frames = 0;
        t.age++;

        bboxes[p.b_idx].set_track_id(t.id);

        spdlog::debug("[tracker] matched track={} age={} bbox=({:.2f},{:.2f},{:.2f}) "
                      "dist={:.3f} speed={:.3f}",
                      t.id, t.age,
                      new_center.x(), new_center.y(), new_center.z(),
                      p.dist, t.velocity.norm());
    }

    // 6. Unmatched bboxes → new tracks.
    for (int b = 0; b < N_bboxes; ++b) {
        if (bbox_matched[b]) continue;

        Track t;
        t.id            = next_track_id_++;
        t.center        = bboxes[b].get_center();
        t.last_bbox     = bboxes[b];
        t.velocity      = Eigen::Vector3d::Zero();
        t.age           = 1;
        t.missed_frames = 0;
        t.dynamic_score = 0.5;   // new tracks start neutral; EMA converges during age guard
        t.is_dynamic    = true;
        t.slow_frames   = 0;

        tracks_.push_back(t);
        bboxes[b].set_track_id(t.id);

        spdlog::debug("[tracker] new track={} at ({:.2f},{:.2f},{:.2f})",
                      t.id, t.center.x(), t.center.y(), t.center.z());
    }

    // 7. Unmatched tracks: increment missed, prune expired.
    for (int t = 0; t < N_tracks; ++t)
        if (!track_matched[t]) tracks_[t].missed_frames++;

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const Track& trk){
                return trk.missed_frames > params_.track_max_missed;
            }),
        tracks_.end());

    spdlog::debug("[tracker] active_tracks={}", tracks_.size());
}

// ===========================================================================
// classify_clusters()
//
// Pipeline:
//   1. Transform cluster_history_ in-place to current sensor frame. [OPT-1]
//   2. update_tracks() on real detections before merge.             [v4-fix]
//   3. merge_with_history() (protected tracks not replaced).
//   4+5. Per-bbox: IoU ratio (step 4, OPT-5), velocity zones (5a),
//        EMA dynamic_score update (5b), age guard.        [OPT-3,OPT-5,FIX-1]
//   6. Push non-phantom bboxes to history.                          [v4-fix]
// ===========================================================================

void DynamicClusterExtractor::classify_clusters(
    Eigen::Isometry3d T_delta_pose,
    std::vector<BoundingBox>& cluster_bboxes)
{
    const Eigen::Isometry3d T_to_current = T_delta_pose.inverse();

    // Fix-2: compute ego-motion magnitude to adaptively scale distance thresholds.
    const double motion      = T_delta_pose.translation().norm();
    const double dist_scale  = 1.0 + params_.motion_scale_factor * motion;

    // Fix-6: Build transformed history on demand using stored poses.
    // Each entry was captured at stored_pose (= T_world_sensor at that time).
    // T_k_to_current = stored_pose * last_pose_.inverse()
    // (consistent with T_to_current = T_world_sensor(t-1) * T_world_sensor(t).inverse())
    // This avoids incremental drift from repeated single-frame transforms.
    std::deque<std::vector<BoundingBox>> history_transformed;
    for (const auto& [stored_pose, frame_bboxes] : cluster_history_) {
        const Eigen::Isometry3d T_k = stored_pose * last_pose_.inverse();
        history_transformed.emplace_back();
        history_transformed.back().reserve(frame_bboxes.size());
        for (const auto& bbox : frame_bboxes) {
            BoundingBox tb = bbox;
            tb.transform(T_k);
            history_transformed.back().push_back(std::move(tb));
        }
    }

    // Step 2: track real detections before merge (with motion-scaled thresholds).
    update_tracks(cluster_bboxes, T_to_current, dist_scale);

    // Collect IDs of established tracks (age > 1) so merge_with_history skips them.
    // Bboxes matched to these tracks represent well-localised, actively-tracked
    // objects.  Replacing them with a phantom from history would give the
    // downstream voxel rejector a stale, static bbox.
    std::unordered_set<int> protected_track_ids;
    for (const auto& t : tracks_)
        if (t.age > 1) protected_track_ids.insert(t.id);

    // Step 3: temporal merge (protects established tracks).
    cluster_bboxes = merge_with_history(cluster_bboxes, history_transformed, protected_track_ids);

    // OPT-3: build track lookup map once for O(1) access in the classification loop.
    std::unordered_map<int, Track*> track_map;
    track_map.reserve(tracks_.size() * 2);
    for (auto& t : tracks_)
        track_map[t.id] = &t;

    // OPT-5: pre-compute squared distance threshold (motion-scaled for Fix-2).
    const double eff_cluster_dist  = params_.cluster_distance_threshold * dist_scale;
    const double dist_thresh2      = eff_cluster_dist * eff_cluster_dist;
    // Fix-5: small tolerance for center-based history match (avoids sole IoU reliance).
    const double center_tol        = params_.containment_margin + 0.1;
    const double center_tol2       = center_tol * center_tol;
    const int available_history = static_cast<int>(history_transformed.size());

    // Steps 4+5: combined overlap-ratio + velocity-zone + EMA classification.
    //
    // For each tracked bbox:
    //   4.  Count overlap matches across all history frames → static_ratio.
    //       (No early break: we need the full count for a meaningful ratio.)
    //   5a. Velocity zone:
    //         < velocity_static_threshold  → static evidence (accumulate slow_frames)
    //         > velocity_dynamic_threshold → dynamic evidence (reset slow_frames)
    //         in between                   → intermediate (decay slow_frames by 1)
    //   5b. EMA update:  score = (1-α)*score + α*observation
    //       where observation = 0.5*overlap_obs + 0.5*vel_obs  (0=static, 1=dynamic)
    //   Age guard: tracks younger than min_track_age are forced DYNAMIC (uncertain).
    for (auto& cluster : cluster_bboxes) {
        const int tid = cluster.get_track_id();
        if (tid < 0) continue;  // phantom — classification inherited from history

        auto it = track_map.find(tid);   // OPT-3: O(1)
        if (it == track_map.end()) {
            cluster.set_dynamic(true);
            continue;
        }
        Track& t = *it->second;

        // --- Step 4: overlap history ratio ---
        // Continuous: overlap_obs = 1.0 - static_ratio (0=fully static, 1=fully dynamic).
        // When history is too short to be meaningful, use a neutral 0.5.
        int static_matches = 0;
        const Eigen::Vector3d& cc = cluster.get_center();
        for (const auto& frame_bboxes : history_transformed) {
            for (const auto& hist_bbox : frame_bboxes) {
                const double d2_h = (cc - hist_bbox.get_center()).squaredNorm();
                if (d2_h > dist_thresh2) continue;
                if (cluster.overlap(hist_bbox) > params_.cluster_iou_threshold ||
                    d2_h < center_tol2) {
                    ++static_matches;
                    break;
                }
            }
        }
        double overlap_obs;
        if (available_history < params_.min_track_age) {
            // Not enough history to draw conclusions — neutral evidence.
            overlap_obs = 0.5;
        } else {
            const double static_ratio = static_cast<double>(static_matches) / available_history;
            overlap_obs = 1.0 - static_ratio;  // continuous: 0=static, 1=dynamic
        }

        // --- Step 5a: velocity zones ---
        const double speed = t.velocity.norm();
        double vel_obs;
        if (speed < params_.velocity_static_threshold) {
            ++t.slow_frames;
            vel_obs = 0.0;
        } else if (speed > params_.velocity_dynamic_threshold) {
            t.slow_frames = 0;
            vel_obs = 1.0;
        } else {
            if (t.slow_frames > 0) --t.slow_frames;
            vel_obs = (speed - params_.velocity_static_threshold) /
                      (params_.velocity_dynamic_threshold - params_.velocity_static_threshold);
        }

        // --- Step 5b: EMA update ---
        // Always update score so it converges even during the age-guard period.
        const double observation = 0.5 * overlap_obs + 0.5 * vel_obs;
        t.dynamic_score = (1.0 - params_.ema_alpha) * t.dynamic_score
                        + params_.ema_alpha * observation;

        // --- Age guard: young tracks output dynamic but EMA already advanced ---
        if (t.age < params_.min_track_age) {
            cluster.set_dynamic(true);
            spdlog::debug("[classifier] track={} age={} < min_track_age={} score={:.3f} -> UNCERTAIN/DYNAMIC",
                          t.id, t.age, params_.min_track_age, t.dynamic_score);
            continue;
        }

        // --- Velocity-based hard static override ---
        // If the track has been consistently slow, force static regardless of score.
        if (t.slow_frames >= params_.velocity_static_frames) {
            t.dynamic_score *= (1.0 - params_.ema_alpha);  // decay toward 0
            t.is_dynamic = false;
            cluster.set_dynamic(false);
            spdlog::debug("[classifier] track={} slow_frames={}>={} -> FORCED STATIC score={:.3f}",
                          t.id, t.slow_frames, params_.velocity_static_frames, t.dynamic_score);
            continue;
        }

        // --- Final EMA decision ---
        t.is_dynamic = (t.dynamic_score >= params_.dynamic_score_threshold);
        cluster.set_dynamic(t.is_dynamic);

        spdlog::debug("[classifier] track={} age={} speed={:.3f} slow_f={} "
                      "overlap_obs={:.2f} vel_obs={:.2f} score={:.3f} -> {}",
                      t.id, t.age, speed, t.slow_frames,
                      overlap_obs, vel_obs,
                      t.dynamic_score,
                      t.is_dynamic ? "DYN" : "STA");
    }

    // Step 6: push non-phantom bboxes to history with current pose.
    // Bboxes are stored in current sensor frame; last_pose_ is T_world_sensor(t).
    std::vector<BoundingBox> history_entry;
    history_entry.reserve(cluster_bboxes.size());
    for (const auto& bbox : cluster_bboxes)
        if (bbox.get_track_id() >= 0)
            history_entry.push_back(bbox);

    cluster_history_.push_front({last_pose_, std::move(history_entry)});
    while (static_cast<int>(cluster_history_.size()) > params_.history_size)
        cluster_history_.pop_back();
}

} // namespace glim