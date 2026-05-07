#include <glim/dynamic_rejection/voxel_filtering.hpp>

#include <algorithm>
#include <cmath>

#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/util/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>
#include <gtsam_points/ann/kdtree.hpp>



#include <spdlog/spdlog.h>

namespace glim {

class PoseKalmanFilter;

WallFilterConfig::WallFilterConfig() {
    spdlog::debug("[wall_filter] WallFilterConfig::WallFilterConfig begin");
    Config config(GlobalConfig::get_config_path("config_wall_filter"));
    Config sensor_config(GlobalConfig::get_config_path("config_sensors"));

    // voxel resolution must match odometry to reuse the same voxelmap
    voxel_resolution        = config.param<double>("wall_filter", "voxel_resolution",        0.5);
    ransac_max_iterations   = config.param<int>   ("wall_filter", "ransac_max_iterations",   500);
    ransac_inlier_threshold = config.param<double>("wall_filter", "ransac_inlier_threshold", 0.15);
    ransac_min_inliers      = config.param<int>   ("wall_filter", "ransac_min_inliers",      8);
    ransac_confidence       = config.param<double>("wall_filter", "ransac_confidence",       0.99);
    wall_vertical_angle_deg = config.param<double>("wall_filter", "wall_vertical_angle_deg", 20.0);
    max_planes              = config.param<int>   ("wall_filter", "max_planes",              8);
    floor_ceiling_angle_deg = config.param<double>("wall_filter", "floor_ceiling_angle_deg", 2.0);
    floor_min_inliers          = config.param<int>   ("wall_filter", "floor_min_inliers",       20);
    wall_bbox_max_aspect_ratio = config.param<double>("wall_filter", "wall_bbox_max_aspect_ratio", 5.0);
    floor_height_threshold      = config.param<double>("wall_filter", "floor_height_threshold", 0.2);

    floor_polar_enabled         = config.param<bool>  ("wall_filter", "floor_polar_enabled",         true);
    floor_polar_r_bin           = config.param<double>("wall_filter", "floor_polar_r_bin",           0.5);
    floor_polar_theta_deg       = config.param<double>("wall_filter", "floor_polar_theta_deg",       10.0);
    floor_polar_slope_threshold = config.param<double>("wall_filter", "floor_polar_slope_threshold", 0.30);
    floor_polar_seed_min_r      = config.param<double>("wall_filter", "floor_polar_seed_min_r",      0.3);
    floor_polar_seed_max_r      = config.param<double>("wall_filter", "floor_polar_seed_max_r",      2.0);
    floor_polar_height_margin   = config.param<double>("wall_filter", "floor_polar_height_margin",   0.30);
    floor_polar_k_lowest        = config.param<int>   ("wall_filter", "floor_polar_k_lowest",        3);
    floor_polar_max_vertical_spread = config.param<double>("wall_filter", "floor_polar_max_vertical_spread", 0.30);



    T_lidar_imu = sensor_config.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
    T_imu_lidar = T_lidar_imu.inverse();
    spdlog::debug("[wall_filter] WallFilterConfig: res={} ransac_iter={} thresh={} min_inliers={} "
                  "conf={} angle_deg={} max_planes={} bbox_max_aspect_ratio={}",
                  voxel_resolution, ransac_max_iterations, ransac_inlier_threshold,
                  ransac_min_inliers, ransac_confidence, wall_vertical_angle_deg, max_planes, wall_bbox_max_aspect_ratio);
}

WallFilterConfig::~WallFilterConfig() {
    spdlog::debug("[wall_filter] WallFilterConfig::~WallFilterConfig");
}

// ---------------------------------------------------------------------------
WallFilter::WallFilter(const WallFilterConfig& config, WallBBoxRegistry::Ptr bbox_registry,const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : config_(config), rng_(std::random_device{}()), bbox_registry_(bbox_registry), pose_kalman_filter_(pose_kalman_filter), last_pose_(Eigen::Isometry3d::Identity()) {}

// ---------------------------------------------------------------------------
WallFilterResult WallFilter::filter(const PreprocessedFrame& frame) {

    WallFilterResult result;

    // ------------------------------------------------------------------
    // Step 1 – Unica voxelizzazione del frame
    // Stesso pattern di DynamicObjectRejectionCPU::dynamic_object_rejection()
    // ------------------------------------------------------------------
    auto voxelmap = std::make_shared<gtsam_points::DynamicVoxelMapCPU>(
        config_.voxel_resolution);

    if (!frame.points.empty()) {
        auto pc = std::make_shared<gtsam_points::PointCloudCPU>(frame.points);
        if (!frame.intensities.empty()) 
            pc->add_intensities(frame.intensities);
        if (!frame.times.empty())       
            pc->add_times(frame.times);
        voxelmap->insert(*pc);
    }

    const int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());

    result.num_total_voxels = nvox;
    result.voxelmap         = voxelmap;

    spdlog::debug("[wall_filter] {} points → {} voxels", frame.points.size(), nvox);

    if (nvox < 3) {
        // Troppo pochi voxel per RANSAC: la voxelmap viene restituita intatta,
        // nessun voxel marcato come parete.
        return result;
    }

    // ------------------------------------------------------------------
    // Step 2 – Estrai centroidi dei voxel
    // ------------------------------------------------------------------
    std::vector<Eigen::Vector3d> centroids = extract_centroids(*voxelmap, nvox);

    // ------------------------------------------------------------------
    // Step 3 – RANSAC iterativo sui centroidi
    // Ad ogni iterazione il piano migliore viene estratto e i suoi inlier
    // rimossi dal set, permettendo di trovare più piani distinti.
    // ------------------------------------------------------------------
    std::vector<PlaneModel> all_planes;
    std::vector<Eigen::Vector3d> remaining = centroids;

    for (int p = 0; p < config_.max_planes; ++p) {
        if (static_cast<int>(remaining.size()) < config_.ransac_min_inliers) break;
        PlaneModel plane;
        if (!ransac_once(remaining, plane)) break;
        all_planes.push_back(plane);
    }

    spdlog::debug("[wall_filter] RANSAC: {} planes extracted", all_planes.size());

    // ------------------------------------------------------------------
    // Step 4 – Classifica i piani come parete o pavimento/soffitto
    // ------------------------------------------------------------------
    for (const auto& plane : all_planes) {
        if (is_wall_plane(plane)) {
            result.wall_planes.push_back(plane);
            spdlog::debug("[wall_filter] wall plane: normal=({:.2f},{:.2f},{:.2f})",
                          plane.normal.x(), plane.normal.y(), plane.normal.z());
        }

        if (is_floor_ceiling_plane(plane)) {
            result.floor_planes.push_back(plane);
            spdlog::debug("[wall_filter] floor plane: normal=({:.2f},{:.2f},{:.2f})",
                          plane.normal.x(), plane.normal.y(), plane.normal.z());
        }
    }


    spdlog::debug("[wall_filter] {} wall planes", result.wall_planes.size());

    if (result.wall_planes.empty() && result.floor_planes.empty()) {
        return result;
    }



    // ------------------------------------------------------------------
    // Step 5 – Marca i voxel-parete nella voxelmap
    // I voxel il cui centroide cade entro ransac_inlier_threshold da un
    // piano-parete vengono marcati con is_wall = true.
    // DynamicObjectRejectionCPU li skippera nel loop di dynamic scoring.
    // ------------------------------------------------------------------
    // result.num_wall_voxels = mark_wall_voxels(*voxelmap, nvox, result.wall_planes);

    spdlog::debug("[wall_filter] marked {}/{} voxels as walls",
                  result.num_wall_voxels, nvox);

    
    if (bbox_registry_) {
        // Reuse the centroid vector already computed above — no need for a second O(nvox) pass.
        const std::vector<Eigen::Vector3d>& all_centroids = centroids;

        std::vector<BoundingBox> wall_bboxes;
        wall_bboxes.reserve(result.wall_planes.size()+result.floor_planes.size());

        for (const auto& plane : result.wall_planes) {
            // 1. Raccogli gli inlier per questo piano
            std::vector<Eigen::Vector3d> inliers;
            for (const auto& c : all_centroids) {
                if (plane.distance(c) < config_.ransac_inlier_threshold)
                    inliers.push_back(c);
            }
            if (static_cast<int>(inliers.size()) < config_.ransac_min_inliers) continue;

            // 2. *** NUOVO: tieni solo il cluster connesso più grande ***
            // eps = 2x la risoluzione voxel (adiacenti sul piano devono connettersi)
            const double cluster_eps = config_.voxel_resolution * 2.5;
            const auto contiguous = extract_largest_contiguous_cluster(inliers, cluster_eps);

            if (static_cast<int>(contiguous.size()) < config_.ransac_min_inliers) {
                spdlog::debug("[wall_filter] plane discarded: largest contiguous cluster "
                            "has only {} inliers", contiguous.size());
                continue;
            }

            // 3. Costruisci la bbox solo sui punti contigui
            BoundingBox bbox = build_wall_bbox(contiguous, plane);

            // 4. *** NUOVO: scarta bbox troppo allungate ***
            if (!is_wall_bbox_compact(bbox)) {
                spdlog::debug("[wall_filter] plane discarded: bbox aspect ratio too large");
                continue;
            }

            wall_bboxes.push_back(bbox);
        }
        spdlog::debug("[wall_filter] built {} wall bboxes",  result.floor_planes.size());

        for (const auto& plane : result.floor_planes) {


            std::vector<Eigen::Vector3d> inliers;
            for (const auto& c : all_centroids) {
                if (plane.distance(c) < config_.ransac_inlier_threshold)
                    inliers.push_back(c);
            }
            if (static_cast<int>(inliers.size()) < config_.floor_min_inliers) continue;

            BoundingBox bbox = build_wall_bbox(inliers, plane);
            wall_bboxes.push_back(bbox);
            
        }

        
        if (!wall_bboxes.empty()) {
            Eigen::Isometry3d T_delta_pose = Eigen::Isometry3d::Identity();
            if (pose_kalman_filter_) {
                
                Eigen::Isometry3d T_world_imu =  pose_kalman_filter_->getPose(); // Inverti per trasformare dal vecchio al nuovo frame
                T_delta_pose = last_pose_.inverse() * T_world_imu; // Delta tra il vecchio e il nuovo frame
                last_pose_ = T_world_imu; // Aggiorna il last_pose_ con il nuovo frame
                spdlog::debug("[wall_filter] obtained delta pose from Kalman filter: translation=({:.4f},{:.4f},{:.4f})",
                              T_delta_pose.translation().x(), T_delta_pose.translation().y(), T_delta_pose.translation().z());

            }
            bbox_registry_->update(wall_bboxes,T_delta_pose.inverse()); // Passa l'inverso del delta pose per trasformare le bbox dal nuovo al vecchio frame
            spdlog::debug("[wall_filter] bbox_registry updated with {} wall OBBs",
                          wall_bboxes.size());
        }
    }

    // ------------------------------------------------------------------
    // Step 5 – Floor voxel classification
    // ------------------------------------------------------------------
    if (config_.floor_polar_enabled) {
        polar_floor_segmentation(*voxelmap, nvox, centroids);
    } else {
        const double lower_z = lowest_point_z(centroids) + config_.floor_height_threshold;
        for (int i = 0; i < nvox; ++i) {
            auto& voxel = voxelmap->lookup_voxel(i);
            if (voxel.mean.z() < lower_z) voxel.is_ground = true;
        }
    }

    // ------------------------------------------------------------------
    // Step 6 – Wall bbox registry: mark voxels inside registered bboxes
    // ------------------------------------------------------------------
    std::vector<bool> empty_bboxes;
    if (bbox_registry_ && !bbox_registry_->bboxes().empty()) {
        empty_bboxes.assign(bbox_registry_->bboxes().size(), true);
        int registry_marked = 0;
        for (int i = 0; i < nvox; ++i) {
            auto& voxel = voxelmap->lookup_voxel(i);
            if (voxel.is_wall) continue;
            for (size_t j = 0; j < bbox_registry_->bboxes().size(); ++j) {
                const auto& bbox = bbox_registry_->bboxes()[j];
                if (bbox.contains(voxel.mean)) {
                    empty_bboxes[j] = false;
                    voxel.is_wall = true;
                    ++registry_marked;
                    break;
                }
            }
        }
        spdlog::debug("[wall_filter] registry marked additional {}/{} voxels as walls",
                      registry_marked, nvox);
        result.num_wall_voxels += registry_marked;
    }



    
    if (bbox_registry_) bbox_registry_->remove_expired(empty_bboxes);
    return result;
}

// ---------------------------------------------------------------------------
std::vector<Eigen::Vector3d> WallFilter::extract_centroids(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox) const
{
    std::vector<Eigen::Vector3d> centroids;
    centroids.reserve(nvox);
    for (int i = 0; i < nvox; ++i) {
        // voxelmap.point(i) restituisce il mean del voxel (Eigen::Vector4d omogeneo)
        auto& voxel = voxelmap.lookup_voxel(i);
        const Eigen::Vector4d& pt = voxel.mean;
        centroids.emplace_back(pt.x(), pt.y(), pt.z());
    }
    return centroids;
}

// ---------------------------------------------------------------------------
PlaneModel WallFilter::fit_plane(const Eigen::Vector3d& p0,
                                  const Eigen::Vector3d& p1,
                                  const Eigen::Vector3d& p2) const
{
    Eigen::Vector3d n    = (p1 - p0).cross(p2 - p0);
    const double    norm = n.norm();

    PlaneModel plane;
    if (norm < 1e-9) {
        // Caso degenere: normale verticale → non sarà classificato parete
        plane.normal = Eigen::Vector3d::UnitZ();
        plane.d      = -plane.normal.dot(p0);
        return plane;
    }
    plane.normal = n / norm;
    plane.d      = -plane.normal.dot(p0);
    return plane;
}

// ---------------------------------------------------------------------------
std::vector<int> WallFilter::find_inliers(
    const std::vector<Eigen::Vector3d>& pts,
    const PlaneModel& plane,
    double threshold) const
{
    std::vector<int> inliers;
    inliers.reserve(pts.size() / 4);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (std::abs(plane.normal.dot(pts[i]) + plane.d) < threshold)
            inliers.push_back(i);
    }
    return inliers;
}

// ---------------------------------------------------------------------------
PlaneModel WallFilter::refit_plane(const std::vector<Eigen::Vector3d>& pts,
                                    const std::vector<int>& inlier_idx) const
{
    const int m = static_cast<int>(inlier_idx.size());

    // Centroide
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int idx : inlier_idx) mean += pts[idx];
    mean /= static_cast<double>(m);

    // Matrice di covarianza
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int idx : inlier_idx) {
        Eigen::Vector3d d = pts[idx] - mean;
        cov += d * d.transpose();
    }

    // Normale = autovettore con autovalore MINIMO (minima varianza perpendicolare)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();

    PlaneModel plane;
    plane.normal = normal;
    plane.d      = -normal.dot(mean);
    return plane;
}

// ---------------------------------------------------------------------------
bool WallFilter::ransac_once(std::vector<Eigen::Vector3d>& pts, PlaneModel& best_plane) {
    const int n = static_cast<int>(pts.size());
    if (n < 3) return false;

    std::uniform_int_distribution<int> dist(0, n - 1);

    int              best_count = 0;
    std::vector<int> best_inliers;

    for (int iter = 0; iter < config_.ransac_max_iterations; ++iter) {
        // Campiona 3 punti distinti
        int i0, i1, i2;
        i0 = dist(rng_);
        do { i1 = dist(rng_); } while (i1 == i0);
        do { i2 = dist(rng_); } while (i2 == i0 || i2 == i1);

        PlaneModel candidate = fit_plane(pts[i0], pts[i1], pts[i2]);
        if (candidate.normal.norm() < 0.5) continue;  // degenerato

        std::vector<int> inliers =
            find_inliers(pts, candidate, config_.ransac_inlier_threshold);

        if (static_cast<int>(inliers.size()) > best_count) {
            best_count   = static_cast<int>(inliers.size());
            best_inliers = std::move(inliers);
            best_plane   = candidate;

            // Terminazione adattiva: N = log(1-p) / log(1-w^3)
            const double w      = static_cast<double>(best_count) / n;
            const double w3     = w * w * w;
            if (w3 > 1.0 - 1e-9) break;
            const double n_iter = std::log(1.0 - config_.ransac_confidence)
                                / std::log(1.0 - w3 + 1e-10);
            if (iter + 1 >= static_cast<int>(std::ceil(n_iter))) break;
        }
    }

    if (best_count < config_.ransac_min_inliers) return false;

    // Refit finale con tutti gli inlier
    best_plane = refit_plane(pts, best_inliers);

    // Rimuovi inlier da pts per la prossima iterazione RANSAC
    std::vector<bool> is_inlier(n, false);
    for (int idx : best_inliers) is_inlier[idx] = true;

    std::vector<Eigen::Vector3d> remaining;
    remaining.reserve(n - best_count);
    for (int i = 0; i < n; ++i)
        if (!is_inlier[i]) remaining.push_back(pts[i]);

    pts = std::move(remaining);
    return true;
}



// Dato un insieme di punti inlier sul piano RANSAC, tieni solo il cluster
// spazialmente connesso più grande (union-find sui centroidi voxel).
std::vector<Eigen::Vector3d> WallFilter::extract_largest_contiguous_cluster(
    const std::vector<Eigen::Vector3d>& inliers,
    double cluster_eps) const
{
    const int n = static_cast<int>(inliers.size());
    if (n == 0) return {};

    // Union-Find
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find = [&](int x) -> int {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    auto unite = [&](int a, int b) {
        parent[find(a)] = find(b);
    };

    // Connetti coppie entro cluster_eps (per n piccolo la O(n²) va bene;
    // per n > ~500 usa un KdTree come nel codice sotto)
    if (n <= 500) {
        const double eps2 = cluster_eps * cluster_eps;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if ((inliers[i] - inliers[j]).squaredNorm() < eps2)
                    unite(i, j);
    } else {
        // KdTree path per dataset grandi
        std::vector<Eigen::Vector4d> pts4(n);
        for (int i = 0; i < n; ++i)
            pts4[i] = Eigen::Vector4d(inliers[i].x(), inliers[i].y(), inliers[i].z(), 1.0);
        gtsam_points::KdTree tree(pts4.data(), n);
        const int K = std::min(n, 16);
        for (int i = 0; i < n; ++i) {
            std::vector<size_t> idx(K);
            std::vector<double> sq(K);
            const size_t found = tree.knn_search(pts4[i].data(), K, idx.data(), sq.data());
            for (size_t k = 0; k < found; ++k)
                if (sq[k] < cluster_eps * cluster_eps)
                    unite(i, static_cast<int>(idx[k]));
        }
    }

    // Conta dimensione di ogni componente
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; ++i)
        components[find(i)].push_back(i);

    int best_root = -1;
    int best_size = 0;
    for (const auto& [root, members] : components) {
        if (static_cast<int>(members.size()) > best_size) {
            best_size = static_cast<int>(members.size());
            best_root = root;
        }
    }

    if (best_root < 0) return inliers;

    std::vector<Eigen::Vector3d> result;
    result.reserve(best_size);
    for (int idx : components[best_root])
        result.push_back(inliers[idx]);

    spdlog::debug("[wall_filter] contiguity: kept {}/{} inliers (largest cluster)",
                  best_size, n);
    return result;
}


// Ritorna false se la bbox è troppo allungata per essere un muro reale.
bool WallFilter::is_wall_bbox_compact(const BoundingBox& bbox) const {
    const Eigen::Vector3d& s = bbox.get_size();
    if (s.minCoeff() < 1e-3) return false;

    // Ordina le dimensioni: minor <= mid <= major
    Eigen::Vector3d dims = s;
    std::sort(dims.data(), dims.data() + 3);
    const double aspect = dims[2] / dims[1];  // major/mid

    // Un muro reale è ragionevolmente rettangolare: rapporto max ~5:1
    return aspect < config_.wall_bbox_max_aspect_ratio;
}

// ---------------------------------------------------------------------------
bool WallFilter::is_wall_plane(const PlaneModel& plane) const {
    // Parete → normale quasi orizzontale → |normal.z| piccolo
    // Pavimento/soffitto → normale quasi verticale → |normal.z| ≈ 1
    //
    // |normal.z| = sin(angolo tra normale e piano XY)
    // Condizione parete: angolo < wall_vertical_angle_deg
    //                  → |normal.z| < sin(wall_vertical_angle_deg)
    const double sin_thr = std::sin(config_.wall_vertical_angle_deg * M_PI / 180.0);
    return std::abs(plane.normal.z()) < sin_thr;
}

bool WallFilter::is_floor_ceiling_plane(const PlaneModel& plane) const {
    // Pavimento/soffitto → normale quasi verticale → |normal.z| vicino a 1
    // Condizione: angolo tra normale e asse Z < floor_ceiling_angle_deg
    //           → |normal.z| > cos(floor_ceiling_angle_deg)
    const double cos_thr = std::cos(config_.floor_ceiling_angle_deg * M_PI / 180.0);
    return std::abs(plane.normal.z()) > cos_thr;
}

// ---------------------------------------------------------------------------
int WallFilter::mark_wall_voxels(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    int nvox,
    const std::vector<PlaneModel>& wall_planes) const
{
    int count = 0;
    for (int i = 0; i < nvox; ++i) {
        auto& voxel = voxelmap.lookup_voxel(i);
        const Eigen::Vector3d c(voxel.mean.x(), voxel.mean.y(), voxel.mean.z());

        for (const auto& plane : wall_planes) {
            if (plane.distance(c) < config_.ransac_inlier_threshold) {
                voxel.is_wall = true;
                ++count;
                break;  // basta un piano
            }
        }
    }
    return count;
}


double WallFilter::lowest_point_z(const std::vector<Eigen::Vector3d>& pts) const {
    double min_z = std::numeric_limits<double>::max();
    for (const auto& p : pts) {
        if (p.z() < min_z) min_z = p.z();
    }
    return min_z;
}

// ---------------------------------------------------------------------------
// Polar ground segmentation.
//
// Converts voxel centroids to (r, θ) polar coords centred on the sensor.
// For each angular sector a ground seed is found at r ∈ [seed_min_r, seed_max_r]
// (the cell with the lowest z), then propagated inward and outward using a
// slope threshold Δz/Δr.  Voxels whose centroid falls in a confirmed ground
// cell and is within height_margin above that cell's z_min are marked
// is_ground = true.
// ---------------------------------------------------------------------------
void WallFilter::polar_floor_segmentation(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    int nvox,
    const std::vector<Eigen::Vector3d>& centroids) const
{
    if (nvox == 0) return;

    const double r_bin         = config_.floor_polar_r_bin;
    const double theta_rad     = config_.floor_polar_theta_deg * M_PI / 180.0;
    const double slope_thr     = config_.floor_polar_slope_threshold;
    const double seed_min_r    = config_.floor_polar_seed_min_r;
    const double seed_max_r    = config_.floor_polar_seed_max_r;
    const double height_margin = config_.floor_polar_height_margin;
    const int    k_lowest      = std::max(1, config_.floor_polar_k_lowest);
    const double max_vertical_spread = config_.floor_polar_max_vertical_spread;
    const double kInf          = std::numeric_limits<double>::max();

    // Grid dimensions
    double max_r = 0.0;
    for (const auto& c : centroids)
        max_r = std::max(max_r, std::sqrt(c.x()*c.x() + c.y()*c.y()));
    max_r += r_bin;

    const int n_theta = static_cast<int>(std::ceil(2.0 * M_PI / theta_rad));
    const int n_r     = static_cast<int>(std::ceil(max_r / r_bin)) + 1;

    // Per-cell statistics
    std::vector<std::vector<double>> z_cell_values(n_theta * n_r);
    std::vector<double> z_grid(n_theta * n_r, kInf);
    std::vector<double> z_min_grid(n_theta * n_r, kInf);
    std::vector<double> z_max_grid(n_theta * n_r, -kInf);
    std::vector<bool>   valid_grid(n_theta * n_r, false);
    std::vector<bool>   ground_grid(n_theta * n_r, false);

    // Per-voxel cell mapping
    std::vector<int> voxel_t(nvox), voxel_r(nvox);

    for (int i = 0; i < nvox; ++i) {
        const auto& c  = centroids[i];
        const double r = std::sqrt(c.x()*c.x() + c.y()*c.y());
        const double theta = std::atan2(c.y(), c.x()) + M_PI; // [0, 2π)

        int r_idx = std::min(static_cast<int>(r / r_bin), n_r - 1);
        int t_idx = static_cast<int>(theta / theta_rad);
        if (t_idx >= n_theta) t_idx = n_theta - 1;

        voxel_t[i] = t_idx;
        voxel_r[i] = r_idx;

        z_cell_values[t_idx * n_r + r_idx].push_back(c.z());
    }

    for (int idx = 0; idx < n_theta * n_r; ++idx) {
        auto& values = z_cell_values[idx];
        if (values.empty()) continue;

        std::sort(values.begin(), values.end());
        const int k = std::min(static_cast<int>(values.size()), k_lowest);
        double sum = 0.0;
        for (int j = 0; j < k; ++j) sum += values[j];

        z_grid[idx] = sum / static_cast<double>(k);
        z_min_grid[idx] = values.front();
        z_max_grid[idx] = values.back();
        valid_grid[idx] = (z_max_grid[idx] - z_min_grid[idx] <= max_vertical_spread);
    }

    const int r_seed_min_idx = static_cast<int>(seed_min_r / r_bin);
    const int r_seed_max_idx = std::min(static_cast<int>(seed_max_r / r_bin), n_r - 1);

    for (int t = 0; t < n_theta; ++t) {
        // Find seed: lowest z in [seed_min_r, seed_max_r]
        double seed_z = kInf;
        int    seed_r = -1;
        for (int r = r_seed_min_idx; r <= r_seed_max_idx; ++r) {
            const int idx = t * n_r + r;
            if (!valid_grid[idx]) continue;
            const double z = z_grid[idx];
            if (z < seed_z) { seed_z = z; seed_r = r; }
        }
        if (seed_r < 0) continue;

        ground_grid[t * n_r + seed_r] = true;

        // Propagate inward (toward sensor)
        double last_z = seed_z;
        int    last_r = seed_r;
        for (int r = seed_r - 1; r >= 0; --r) {
            const int idx = t * n_r + r;
            if (!valid_grid[idx]) continue; // empty or invalid vertical spread cell
            const double z = z_grid[idx];
            const double dr = (last_r - r) * r_bin;
            if (dr <= 1e-6) continue;
            if (std::abs(z - last_z) / dr < slope_thr) {
                ground_grid[idx] = true;
                last_z = z; last_r = r;
            } else { break; }
        }

        // Propagate outward (away from sensor)
        last_z = seed_z; last_r = seed_r;
        for (int r = seed_r + 1; r < n_r; ++r) {
            const int idx = t * n_r + r;
            if (!valid_grid[idx]) continue;
            const double z = z_grid[idx];
            const double dr = (r - last_r) * r_bin;
            if (std::abs(z - last_z) / dr < slope_thr) {
                ground_grid[idx] = true;
                last_z = z; last_r = r;
            } else { break; }
        }
    }

    // Mark floor voxels
    int count = 0;
    for (int i = 0; i < nvox; ++i) {
        if (!ground_grid[voxel_t[i] * n_r + voxel_r[i]]) continue;
        const double ground_z = z_grid[voxel_t[i] * n_r + voxel_r[i]];
        if (centroids[i].z() <= ground_z + height_margin) {
            voxelmap.lookup_voxel(i).is_ground = true;
            ++count;
        }
    }
    spdlog::debug("[wall_filter] polar floor: {}/{} voxels classified as ground", count, nvox);
}

BoundingBox WallFilter::build_wall_bbox(
    const std::vector<Eigen::Vector3d>& inlier_pts,
    const PlaneModel& plane) const
{
    if (inlier_pts.empty()) {
        return BoundingBox(Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Matrix3d::Identity());
    }

    // Usa direttamente gli assi del piano senza normalizzare verso UnitZ
    const Eigen::Vector3d n = plane.normal.normalized();

    // Asse arbitrario non parallelo a n per costruire il frame
    Eigen::Vector3d ref = Eigen::Vector3d::UnitX();
    if (std::abs(n.dot(ref)) > 0.9) {
        ref = Eigen::Vector3d::UnitY();
    }

    // Gram-Schmidt: right ortogonale a n, up ortogonale a entrambi
    const Eigen::Vector3d right = (ref - n.dot(ref) * n).normalized();
    const Eigen::Vector3d up    = n.cross(right).normalized();

    Eigen::Matrix3d R;
    R.col(0) = n;
    R.col(1) = right;
    R.col(2) = up;

    // ── Proietta tutti i centroidi nel frame locale ───────────────────────────
    // Usiamo il primo punto come origine temporanea per stabilità numerica
    const Eigen::Vector3d origin = inlier_pts[0];

    Eigen::Vector3d local_min( 1e9,  1e9,  1e9);
    Eigen::Vector3d local_max(-1e9, -1e9, -1e9);

    for (const auto& p : inlier_pts) {
        const Eigen::Vector3d local = R.transpose() * (p - origin);
        local_min = local_min.cwiseMin(local);
        local_max = local_max.cwiseMax(local);
    }

    // ── Dimensioni ────────────────────────────────────────────────────────────
    Eigen::Vector3d size = local_max - local_min;

    // Lo spessore lungo la normale (asse 0) è artificialmente piccolo
    // perché i centroidi sono quasi complanari: imponiamo un minimo
    size.x() = std::max(size.x(), config_.ransac_inlier_threshold * 2.0);

    // ── Centro nel frame locale → mondo ──────────────────────────────────────
    const Eigen::Vector3d local_center = 0.5 * (local_max + local_min);
    const Eigen::Vector3d world_center = R * local_center + origin;

    return BoundingBox(size, world_center, R);
}


}  // namespace glim