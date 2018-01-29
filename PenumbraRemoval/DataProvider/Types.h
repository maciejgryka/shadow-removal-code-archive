#ifndef DATA_PROVIDER_TYPES_H
#define DATA_PROVIDER_TYPES_H

#include <utility>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

// Eigen dynamic matrix of floats stored in row-major order
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMat;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenMatCM;

// offset of a rectangle by translation (x, y) and rotation angle
// typedef std::pair<cv::Point2i, float> RotatedRectOffset;
class RotatedRectOffset {
public:
  RotatedRectOffset():
      rot_rect_(cv::Point2f(-1.f, -1.f), cv::Size2f(-1.f, -1.f), -1.f),
      intensity_offset_(-1.f) {}

  RotatedRectOffset(const cv::Point2f& center, const float& angle, const float& intensity_offset):
      rot_rect_(center, cv::Size2f(-1.f, -1.f), angle),
      intensity_offset_(intensity_offset) {}

  RotatedRectOffset(const cv::RotatedRect& rot_rect, const float& intensity_offset):
      rot_rect_(rot_rect),
      intensity_offset_(intensity_offset) {}

  ~RotatedRectOffset() {}

  float angle() const { return rot_rect_.angle; }

  const cv::Point2f& offset() const { return rot_rect_.center; }

  void set_angle(float angle) { rot_rect_.angle = angle; }

  void set_offset(const cv::Point2f& offset) { rot_rect_.center = offset; }

  float intensity_offset() const { return intensity_offset_; }

  void set_intensity_offset(float int_offset) { intensity_offset_ = int_offset; }

private:
  cv::RotatedRect rot_rect_;
  float intensity_offset_;
};

#endif // DATA_PROVIDER_TYPES_H
