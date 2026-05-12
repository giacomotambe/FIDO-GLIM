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
    std::vector<BoundingBox> get_bounding_boxes() const { return bboxes_; }
    void set_bounding_boxes(const std::vector<BoundingBox>& bboxes) { bboxes_ = bboxes; }
    PreprocessedFrame::Ptr get_last_dynamic_frame() const { return last_dynamic_frame; }

    const VelocityInflationParams& get_inflate_params() const { return inflate_params_; }
private:
    std::vector<int> find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const;
private:
    std::vector<BoundingBox> bboxes_;
    std::vector<int>         bbox_ages_;   ///< frame age of each bbox; removed when age > max_bbox_frames_
    int                      max_bbox_frames_;
    PreprocessedFrame::Ptr last_dynamic_frame = nullptr;
    double inflate_margin_;
    VelocityInflationParams inflate_params_;
};  

}  // namespace glim    