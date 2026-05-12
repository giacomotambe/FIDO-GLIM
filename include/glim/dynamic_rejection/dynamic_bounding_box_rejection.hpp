#include <memory>
#include <vector>
#include <Eigen/Geometry>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>

namespace glim {
class DynamicBBoxRejection {
public:
    using Ptr = std::shared_ptr<DynamicBBoxRejection>;
    using ConstPtr = std::shared_ptr<const DynamicBBoxRejection>;

    explicit DynamicBBoxRejection(const std::vector<BoundingBox>& bbox = {});
    ~DynamicBBoxRejection();

    /**
     * @brief Classify points as dynamic if they are outside the bounding box and return a new estimation frame without points classified as dynamic.
     * @param frame Current preprocessed frame
     * @return New preprocessed frame with dynamic points removed
     */
    PreprocessedFrame::Ptr reject(const PreprocessedFrame::Ptr frame);  
    void insert_bounding_boxes(BoundingBox& bbox);
    void clear_bounding_boxes() { bboxes_.clear(); }
    void set_bounding_boxes(const std::vector<BoundingBox>& bboxes) { bboxes_ = bboxes; }
    PreprocessedFrame::Ptr get_last_dynamic_frame() const { return last_dynamic_frame; }

    double get_v_fwd_k()  const { return v_fwd_k_; }
    double get_v_rear_k() const { return v_rear_k_; }
    double get_v_lat_k()  const { return v_lat_k_; }
    double get_v_min()    const { return v_min_; }
private:
    std::vector<int> find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const;
private:
    std::vector<BoundingBox> bboxes_;
    PreprocessedFrame::Ptr last_dynamic_frame = nullptr;
    double inflate_margin_;
    // Velocity-ellipsoid inflation parameters (loaded from config_bbox_rejection)
    double v_fwd_k_;   ///< Forward semi-axis multiplier  (base + speed * k)
    double v_rear_k_;  ///< Rear semi-axis multiplier
    double v_lat_k_;   ///< Lateral semi-axis multiplier
    double v_min_;     ///< Minimum XY speed [m/s] to activate ellipsoid inflation
};  

}  // namespace glim    