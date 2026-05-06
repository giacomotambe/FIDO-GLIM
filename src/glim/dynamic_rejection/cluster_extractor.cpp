/*
    FILE: cluster_extractor.cpp
    ------------------
    Dynamic cluster extractor: DBSCAN → NMS → peer merge → overlap tracking.

    Pipeline:
      1. cluster_voxels()       — DBSCAN on non-wall voxel centroids
      2. build_point_clusters() — collect raw points per cluster
      3. compute_bounding_boxes() + nms3D() — AABB per cluster, remove duplicates
      4. merge_nearby_clusters() — iteratively merge bboxes closer than peer_merge_distance
      5. update_tracks()        — overlap-based tracking (no velocity estimation)
*/

#include <glim/dynamic_rejection/cluster_extractor.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
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

    cluster_iou_threshold = config.param<double>("dynamic_cluster_extractor", "cluster_iou_threshold", 0.5);

    peer_merge_distance = config.param<double>("dynamic_cluster_extractor", "peer_merge_distance", 0.0);

    track_match_distance = config.param<double>("dynamic_cluster_extractor", "track_match_distance", 1.5);
    track_match_iou      = config.param<double>("dynamic_cluster_extractor", "track_match_iou",      0.3);
    track_max_missed     = config.param<int>   ("dynamic_cluster_extractor", "track_max_missed",     3);
    min_dynamic_frames   = config.param<int>   ("dynamic_cluster_extractor", "min_dynamic_frames",   3);

    spdlog::debug("[cluster_extractor] eps_factor={:.2f} min_pts={} knn_max={} "
                  "min_cluster_voxels={} min_points_bbox={} "
                  "bbox_extent=[{:.2f},{:.2f}] bbox_volume=[{:.3f},{:.3f}] "
                  "cluster_iou_threshold={:.2f} peer_merge_distance={:.2f} "
                  "track_match_distance={:.2f} track_match_iou={:.2f} track_max_missed={} "
                  "min_dynamic_frames={}",
                  eps_voxel_factor, min_pts, knn_max_neighbors,
                  min_cluster_voxels, min_points_for_bbox,
                  bbox_min_extent, bbox_max_extent,
                  bbox_min_volume, bbox_max_volume,
                  cluster_iou_threshold, peer_merge_distance,
                  track_match_distance, track_match_iou, track_max_missed,
                  min_dynamic_frames);
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

    bboxes = merge_nearby_clusters(bboxes);

    spdlog::debug("[cluster_extractor] {} bboxes after peer merge", bboxes.size());

    if (pose_kalman_filter_) {
        const Eigen::Isometry3d cur_pose      = pose_kalman_filter_->getPose();
        const Eigen::Isometry3d T_to_current  = (cur_pose * last_pose_.inverse()).inverse();
        last_pose_ = cur_pose;
        update_tracks(bboxes, T_to_current);
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
        if (!createAABB(cluster, bbox)) {
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
// createAABB()  —  AABB (axis-aligned, single pass over points)
// ===========================================================================

bool DynamicClusterExtractor::createAABB(
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
// compute_union_bbox()  —  union AABB of two bboxes, keeping a's local frame
// ===========================================================================

static BoundingBox compute_union_bbox(const BoundingBox& a, const BoundingBox& b)
{
    const Eigen::Matrix3d& R    = a.get_rotation();
    const Eigen::Vector3d& c_a  = a.get_center();
    const Eigen::Vector3d  he_a = a.get_size() * 0.5;

    const Eigen::Vector3d c_b_local = R.transpose() * (b.get_center() - c_a);
    const Eigen::Matrix3d R_b_local = R.transpose() * b.get_rotation();
    const Eigen::Vector3d he_b      = b.get_size() * 0.5;

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
// merge_nearby_clusters()
//
// Iteratively merge current-frame bboxes whose centers are within
// peer_merge_distance.  Converges when no more merges happen (handles chains
// A–B–C where A and C only become close after B is absorbed into A).
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::merge_nearby_clusters(
    const std::vector<BoundingBox>& bboxes) const
{
    if (params_.peer_merge_distance <= 0.0 || bboxes.empty()) return bboxes;

    const double d2_max = params_.peer_merge_distance * params_.peer_merge_distance;
    std::vector<BoundingBox> working(bboxes);

    bool any_merge = true;
    while (any_merge) {
        any_merge = false;
        const int W = static_cast<int>(working.size());

        // Process largest bboxes first so small ones are absorbed into larger ones.
        std::vector<int> order(W);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return working[a].get_size().prod() > working[b].get_size().prod();
        });

        std::vector<bool>        absorbed(W, false);
        std::vector<Eigen::Vector3d> centers(W);
        for (int i = 0; i < W; ++i) centers[i] = working[i].get_center();

        for (int ii = 0; ii < W; ++ii) {
            const int i = order[ii];
            if (absorbed[i]) continue;
            for (int jj = ii + 1; jj < W; ++jj) {
                const int j = order[jj];
                if (absorbed[j]) continue;
                if ((centers[i] - centers[j]).squaredNorm() <= d2_max) {
                    working[i] = compute_union_bbox(working[i], working[j]);
                    centers[i] = working[i].get_center();
                    absorbed[j] = true;
                    any_merge   = true;
                }
            }
        }

        std::vector<BoundingBox> tmp;
        tmp.reserve(W);
        for (int i = 0; i < W; ++i)
            if (!absorbed[i]) tmp.push_back(working[i]);
        working = std::move(tmp);
    }

    spdlog::debug("[merge_nearby_clusters] {} -> {} bboxes", bboxes.size(), working.size());
    return working;
}

// ===========================================================================
// update_tracks()
//
// Overlap-based tracking: no velocity estimation.
// Matching criterion: center distance < track_match_distance (cheap gate)
//                     AND overlap > track_match_iou.
// Greedy assignment ranked by overlap (best overlap first).
// ===========================================================================

void DynamicClusterExtractor::update_tracks(
    std::vector<BoundingBox>& bboxes,
    const Eigen::Isometry3d&  T_to_current)
{
    const int N_tracks = static_cast<int>(tracks_.size());
    const int N_bboxes = static_cast<int>(bboxes.size());

    // Transform track centers into current sensor frame.
    for (auto& t : tracks_)
        t.center = T_to_current * t.center;

    // Build candidate pairs gated by distance, ranked by overlap.
    struct Pair { int t_idx, b_idx; double overlap; };
    std::vector<Pair> pairs;

    const double D2 = params_.track_match_distance * params_.track_match_distance;

    for (int t = 0; t < N_tracks; ++t) {
        for (int b = 0; b < N_bboxes; ++b) {
            if ((tracks_[t].center - bboxes[b].get_center()).squaredNorm() > D2) continue;
            const double ov = tracks_[t].last_bbox.overlap(bboxes[b]);
            if (ov >= params_.track_match_iou)
                pairs.push_back({t, b, ov});
        }
    }

    // Best overlap first for greedy assignment.
    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b){ return a.overlap > b.overlap; });

    std::vector<bool> track_matched(N_tracks, false);
    std::vector<bool> bbox_matched (N_bboxes, false);

    for (const auto& p : pairs) {
        if (track_matched[p.t_idx] || bbox_matched[p.b_idx]) continue;
        track_matched[p.t_idx] = true;
        bbox_matched [p.b_idx] = true;

        Track& t        = tracks_[p.t_idx];
        t.center        = bboxes[p.b_idx].get_center();
        // Gate is_dynamic on the hysteresis counter updated by update_dynamic_feedback().
        // A bbox is flagged dynamic only after min_dynamic_frames consecutive confirmations.
        bboxes[p.b_idx].set_dynamic(t.dynamic_frames >= params_.min_dynamic_frames);
        t.last_bbox     = bboxes[p.b_idx];
        t.missed_frames = 0;
        t.age++;
        bboxes[p.b_idx].set_track_id(t.id);
       
        spdlog::debug("[tracker] matched track={} age={} overlap={:.3f} center=({:.2f},{:.2f},{:.2f})",
                      t.id, t.age, p.overlap,
                      t.center.x(), t.center.y(), t.center.z());
    }

    // Unmatched bboxes → new tracks.
    for (int b = 0; b < N_bboxes; ++b) {
        if (bbox_matched[b]) continue;
        Track t;
        t.id            = next_track_id_++;
        t.center        = bboxes[b].get_center();
        t.last_bbox     = bboxes[b];
        t.age           = 1;
        t.missed_frames = 0;
        tracks_.push_back(t);
        bboxes[b].set_track_id(t.id);
        bboxes[b].set_dynamic(false);  // Mark tracks as dynamic for downstream use.
        spdlog::debug("[tracker] new track={} at ({:.2f},{:.2f},{:.2f})",
                      t.id, t.center.x(), t.center.y(), t.center.z());
    }

    // Increment missed frames and prune dead tracks.
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
// update_dynamic_feedback()
//
// Must be called after reject() each frame.  Matches post-rejection bboxes to
// tracks by track_id and updates the dynamic_frames hysteresis counter:
//   - bbox confirmed dynamic  → increment counter
//   - bbox not dynamic / not found → reset counter to 0
// ===========================================================================

void DynamicClusterExtractor::update_dynamic_feedback(
    const std::vector<BoundingBox>& post_rejection_bboxes)
{
    for (auto& track : tracks_) {
        bool found = false;
        for (const auto& bbox : post_rejection_bboxes) {
            if (bbox.get_track_id() != track.id) continue;
            track.dynamic_frames = bbox.is_dynamic_bbox() ? track.dynamic_frames + 1 : 0;
            found = true;
            break;
        }
        if (!found) {
            track.dynamic_frames = 0;
        }
        spdlog::debug("[tracker] track={} dynamic_frames={}", track.id, track.dynamic_frames);
    }
}

} // namespace glim
