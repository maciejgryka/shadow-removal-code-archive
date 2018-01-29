#ifndef ALIGNED_PATCH_H
#define ALIGNED_PATCH_H

#include <opencv2/core/core.hpp>

class AlignedPatch {
public:
  AlignedPatch():
      rot_rect_(cv::Point2f(-1.f, -1.f), cv::Size2f(-1.f, -1.f), -1.f) {}
  
  explicit AlignedPatch(const cv::RotatedRect& rot_rect, const float& intensity_offset = 0.f):
      rot_rect_(rot_rect),
      intensity_offset_(intensity_offset) {}

  ~AlignedPatch() {}
  
  const cv::RotatedRect& rot_rect() const {
    return rot_rect_;
  }

  void set_rot_rect(const cv::RotatedRect& rot_rect) {
    rot_rect_ = rot_rect;
  }

  cv::RotatedRect* rot_rect_ptr() {
    return &rot_rect_;
  }

  float intensity_offset() const {
    return intensity_offset_;
  }

  void set_intensity_offset(const float& intensity_offset) {
    intensity_offset_ = intensity_offset;
  }

private:
  cv::RotatedRect rot_rect_;
  float intensity_offset_;
};

#endif // ALIGNED_PATCH_H