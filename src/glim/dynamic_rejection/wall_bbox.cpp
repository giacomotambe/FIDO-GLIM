#include <glim/dynamic_rejection/wall_bbox.hpp>
#include <spdlog/spdlog.h>
#include <glim/util/config.hpp>
#include <Eigen/Eigenvalues>
namespace glim {


WallBBoxRegistryConfig::WallBBoxRegistryConfig() {
    spdlog::debug("[wall_registry] WallBBoxRegistryConfig::WallBBoxRegistryConfig");
    Config config(GlobalConfig::get_config_path("config_wall_registry"));

    overlap_threshold = config.param<double>("wall_registry", "overlap_threshold", 0.3);
    merge_weight     = config.param<double>("wall_registry", "merge_weight",     0.3);
    enable_expiry    = config.param<bool>  ("wall_registry", "enable_expiry",    false);
    max_missed_frames= config.param<int>   ("wall_registry", "max_missed_frames", 30);
    min_normal_dot   = config.param<double>("wall_registry", "min_normal_dot",   0.9);
    max_center_distance = config.param<double>("wall_registry", "max_center_distance", 5.0);

    spdlog::debug("[wall_registry] WallBBoxRegistryConfig: overlap_thresh={} merge_weight={} "
                  "enable_expiry={} max_missed_frames={}",
                  overlap_threshold, merge_weight, enable_expiry, max_missed_frames);
}


WallBBoxRegistry::WallBBoxRegistry(const WallBBoxRegistryConfig& config)
    : config_(config) {
        registry_.clear();
        missed_frames_.clear();
    }


WallBBoxRegistryConfig::~WallBBoxRegistryConfig() {
    spdlog::debug("[wall_registry] WallBBoxRegistryConfig::~WallBBoxRegistryConfig");
}
// ---------------------------------------------------------------------------
// update()
// ---------------------------------------------------------------------------

void WallBBoxRegistry::update(const std::vector<BoundingBox>& new_bboxes , const Eigen::Isometry3d& delta_pose) {
    spdlog::debug("[wall_registry] update: new_bboxes={}, delta_pose translation=({:.3f},{:.3f},{:.3f})",
                  new_bboxes.size(), delta_pose.translation().x(), delta_pose.translation().y(), delta_pose.translation().z());
    
    transform_existing_bboxes(delta_pose);
    const int n_existing = static_cast<int>(registry_.size());
    std::vector<bool> matched(n_existing, false);

    for (const auto& incoming : new_bboxes) {
        if (incoming.get_size() == Eigen::Vector3d::Zero()) continue;

        int    best_idx = -1;
        double best_iou  = config_.overlap_threshold;

        for (int i = 0; i < n_existing; ++i) {

            // --- [NUOVO] Controlla che le due bbox abbiano la stessa normale
            // La colonna 0 della rotation matrix è la normale al piano
            const Eigen::Vector3d n_reg      = registry_[i].get_rotation().col(0);
            const Eigen::Vector3d n_incoming = incoming.get_rotation().col(0);
            const double normal_dot = std::abs(n_reg.dot(n_incoming));

            // Se le normali sono troppo diverse, non sono lo stesso muro
            if (normal_dot < config_.min_normal_dot) {
                spdlog::debug("[wall_registry] skip merge: normal mismatch dot={:.3f}", normal_dot);
                continue;
            }

            // --- [NUOVO] Controlla che i centri non siano troppo lontani
            const double center_dist = (registry_[i].get_center() - incoming.get_center()).norm();
            if (center_dist > config_.max_center_distance) {
                spdlog::debug("[wall_registry] skip merge: centers too far {:.3f}m", center_dist);
                continue;
            }

            const double overlap = iou(registry_[i], incoming);
            if (overlap > best_iou) {
                best_iou  = overlap;
                best_idx  = i;
            }
        }
        if (best_idx >= 0) {
            registry_[best_idx] = merge(registry_[best_idx], incoming, config_.merge_weight);
            matched[best_idx]   = true;
            missed_frames_[best_idx] = 0;
            spdlog::debug("[wall_registry] merged bbox {}: IoU={:.3f}", best_idx, best_iou);

        } else {
            registry_.push_back(incoming);
            missed_frames_.push_back(0);
            spdlog::debug("[wall_registry] added new bbox (total={})", registry_.size());
        }
    }

    if (config_.enable_expiry) {
        // Only count missed frames for bboxes that existed at the start of this
        // update — newly added ones (index >= n_existing) were just seen now.
        for (int i = 0; i < n_existing; ++i)
            if (!matched[i]) ++missed_frames_[i];

        for (int i = static_cast<int>(registry_.size()) - 1; i >= 0; --i) {
            if (missed_frames_[i] > config_.max_missed_frames) {
                spdlog::debug("[wall_registry] expired bbox {}", i);
                registry_.erase(registry_.begin() + i);
                missed_frames_.erase(missed_frames_.begin() + i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OBB helpers
// ---------------------------------------------------------------------------

WallBBoxRegistry::OBB WallBBoxRegistry::to_obb(const BoundingBox& bbox) {
    // BoundingBox espone: size (full extents), center, rotation (col = assi)
    // Dobbiamo accedere ai campi privati tramite un helper pubblico.
    // Poiché BoundingBox non ha getter, ricostruiamo l'OBB dall'interfaccia
    // contains() — ma è più pulito aggiungere getter a BoundingBox.
    // Per ora usiamo la convenzione: BoundingBox(size, center, rotation)
    // e accediamo ai dati tramite i parametri del costruttore memorizzati
    // nella struttura OBB che costruiamo qui.
    //
    // NOTA: questo richiede che BoundingBox esponga i getter size/center/rotation.
    // Se non li espone, aggiungili a bounding_box.hpp:
    //   const Eigen::Vector3d& get_size()     const { return size; }
    //   const Eigen::Vector3d& get_center()   const { return center; }
    //   const Eigen::Matrix3d& get_rotation() const { return rotation; }
    OBB obb;
    obb.center       = bbox.get_center();
    obb.axes         = bbox.get_rotation();           // colonne = assi locali
    obb.half_extents = bbox.get_size() * 0.5;
    return obb;
}

double WallBBoxRegistry::project_obb(const OBB& obb, const Eigen::Vector3d& axis) {
    // Raggio della proiezione dell'OBB sull'asse
    return obb.half_extents[0] * std::abs(obb.axes.col(0).dot(axis))
         + obb.half_extents[1] * std::abs(obb.axes.col(1).dot(axis))
         + obb.half_extents[2] * std::abs(obb.axes.col(2).dot(axis));
}

double WallBBoxRegistry::interval_overlap(double c1, double r1,
                                           double c2, double r2) {
    return std::max(0.0, std::min(c1 + r1, c2 + r2) - std::max(c1 - r1, c2 - r2));
}

double WallBBoxRegistry::intersection_volume(const OBB& a, const OBB& b) {
    // SAT con 15 assi separatori: 3 di A, 3 di B, 9 prodotti vettoriali
    // Per ogni asse calcoliamo la sovrapposizione dei due intervalli proiettati.
    // Il volume di intersezione è approssimato come prodotto delle 3 sovrapposizioni
    // lungo gli assi di A (approssimazione valida per box quasi allineate).

    const Eigen::Vector3d t = b.center - a.center;

    // Proiezioni del vettore tra centri sui 15 assi
    // e sovrapposizioni corrispondenti
    double overlap[15];
    Eigen::Vector3d axes[15];

    int k = 0;
    for (int i = 0; i < 3; ++i) axes[k++] = a.axes.col(i);
    for (int i = 0; i < 3; ++i) axes[k++] = b.axes.col(i);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            axes[k] = a.axes.col(i).cross(b.axes.col(j));
            if (axes[k].norm() > 1e-6) axes[k].normalize();
            ++k;
        }

    for (int i = 0; i < 15; ++i) {
        if (axes[i].norm() < 1e-6) { overlap[i] = 1e9; continue; }
        const double center_proj = std::abs(t.dot(axes[i]));
        const double ra = project_obb(a, axes[i]);
        const double rb = project_obb(b, axes[i]);
        overlap[i] = std::max(0.0, ra + rb - center_proj);
        if (overlap[i] <= 0.0) return 0.0;  // asse separatore trovato
    }

    // Approssima il volume di intersezione con il prodotto delle prime 3 sovrapposizioni
    // (quelle lungo gli assi di A), clampato al volume reale di A
    const double vol_intersection =
        std::min(overlap[0], 2.0 * a.half_extents[0]) *
        std::min(overlap[1], 2.0 * a.half_extents[1]) *
        std::min(overlap[2], 2.0 * a.half_extents[2]);

    return vol_intersection;
}

double WallBBoxRegistry::iou(const BoundingBox& a, const BoundingBox& b) {
    const OBB oa = to_obb(a);
    const OBB ob = to_obb(b);

    const double vol_a = 8.0 * oa.half_extents.prod();
    const double vol_b = 8.0 * ob.half_extents.prod();

    if (vol_a < 1e-9 || vol_b < 1e-9) return 0.0;

    const double vol_inter = intersection_volume(oa, ob);
    return vol_inter / (vol_a + vol_b - vol_inter + 1e-9);
}


void WallBBoxRegistry::transform_existing_bboxes(const Eigen::Isometry3d& delta_pose) {
    for (auto& bbox : registry_) {
        bbox.transform(delta_pose);
    }
}

// ---------------------------------------------------------------------------
// merge()
// ---------------------------------------------------------------------------

BoundingBox WallBBoxRegistry::merge(const BoundingBox& existing,
                                     const BoundingBox& incoming,
                                     double weight)
{
    // Manteniamo il frame locale della bbox esistente (rotazione fissa)
    const Eigen::Matrix3d& R    = existing.get_rotation();
    const Eigen::Vector3d& c_ex = existing.get_center();

    // Half extents della bbox esistente nel suo frame locale
    const Eigen::Vector3d he_ex = existing.get_size() * 0.5;

    // Trasforma il centro della incoming nel frame locale della existing
    const Eigen::Vector3d c_in_local = R.transpose() * (incoming.get_center() - c_ex);
    const Eigen::Vector3d he_in      = incoming.get_size() * 0.5;

    // Gli 8 vertici della incoming nel frame locale della existing
    // (ruotiamo anche l'orientamento della incoming nel frame della existing)
    const Eigen::Matrix3d R_in_local = R.transpose() * incoming.get_rotation();

    Eigen::Vector3d local_min = -he_ex;
    Eigen::Vector3d local_max =  he_ex;

    for (int sx : {-1, 1})
    for (int sy : {-1, 1})
    for (int sz : {-1, 1}) {
        const Eigen::Vector3d corner_local =
            R_in_local * Eigen::Vector3d(sx * he_in.x(),
                                          sy * he_in.y(),
                                          sz * he_in.z());
        const Eigen::Vector3d v = c_in_local + corner_local;
        local_min = local_min.cwiseMin(v);
        local_max = local_max.cwiseMax(v);
    }

    // Nuove dimensioni e centro nel mondo
    const Eigen::Vector3d new_size   = local_max - local_min;
    const Eigen::Vector3d new_center = c_ex + R * (0.5 * (local_max + local_min));

    return BoundingBox(new_size, new_center, R);
}


// BoundingBox WallBBoxRegistry::merge(const BoundingBox& existing,
//                                      const BoundingBox& incoming,
//                                      double weight)
// {
//     // Keep existing frame
//     const Eigen::Matrix3d& R    = existing.get_rotation();
//     const Eigen::Vector3d& c_ex = existing.get_center();

//     // Half extents
//     const Eigen::Vector3d he_ex = existing.get_size() * 0.5;
//     const Eigen::Vector3d he_in = incoming.get_size() * 0.5;

//     // Transform incoming center into existing local frame
//     const Eigen::Vector3d c_in_local =
//         R.transpose() * (incoming.get_center() - c_ex);

//     // Incoming rotation in existing frame
//     const Eigen::Matrix3d R_in_local =
//         R.transpose() * incoming.get_rotation();

//     // Initialize bounds with existing box
//     Eigen::Vector3d local_min = -he_ex;
//     Eigen::Vector3d local_max =  he_ex;

//     // Expand bounds with incoming corners
//     for (int sx : {-1, 1})
//     for (int sy : {-1, 1})
//     for (int sz : {-1, 1}) {
//         Eigen::Vector3d corner_local =
//             R_in_local * Eigen::Vector3d(sx * he_in.x(),
//                                         sy * he_in.y(),
//                                         sz * he_in.z());

//         Eigen::Vector3d v = c_in_local + corner_local;

//         if (!v.allFinite()) continue;

//         local_min = local_min.cwiseMin(v);
//         local_max = local_max.cwiseMax(v);
//     }

//     // New size from geometric merge
//     Eigen::Vector3d new_size = local_max - local_min;

//     // Existing size
//     const Eigen::Vector3d size_ex = existing.get_size();

//     // Identify thickness axis (smallest dimension of existing box)
//     int thickness_axis;
//     size_ex.minCoeff(&thickness_axis);

//     // Extract thickness values
//     const double thickness_ex = size_ex[thickness_axis];
//     const double thickness_new = new_size[thickness_axis];

//     // Accept update only if consistent (avoid sudden expansion)
//     const double max_ratio = 2.0; // tunable
//     bool accept_update =
//         (thickness_new > 0.0) &&
//         (thickness_new < thickness_ex * max_ratio);

//     if (accept_update) {
//         // Weighted update of thickness
//         new_size[thickness_axis] =
//             (1.0 - weight) * thickness_ex +
//             weight * thickness_new;
//     } else {
//         // Reject update, keep previous thickness
//         new_size[thickness_axis] = thickness_ex;
//     }

//     // Compute center in local frame
//     Eigen::Vector3d center_local = 0.5 * (local_max + local_min);

//     // Stabilize center along thickness axis (avoid drifting)
//     center_local[thickness_axis] = 0.0;

//     // Transform back to world
//     const Eigen::Vector3d new_center = c_ex + R * center_local;

//     // Final safety check
//     if (!new_center.allFinite() || !new_size.allFinite()) {
//         return existing;
//     }

//     return BoundingBox(new_size, new_center, R);
// }



void WallBBoxRegistry::remove_expired(const std::vector<bool>& empty_bboxes) {
    for (int i = static_cast<int>(registry_.size()) - 1; i >= 0; --i) {
        if (empty_bboxes[i]) {
            spdlog::debug("[wall_registry] clearing empty bbox {}", i);
            registry_.erase(registry_.begin() + i);
            missed_frames_.erase(missed_frames_.begin() + i);
        }
    }
} 
} // namespace glim