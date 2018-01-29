#ifndef DATA_SEARCHER_H
#define DATA_SEARCHER_H

#include <vector>

#include <DataProvider/DataProvider.h>
#include <DataProvider/PCAWrapper.h>

#include "Types.h"

using cv::RotatedRect;

bool comparator2(const FloatInt& lhs, const FloatInt& rhs) { return lhs.first < rhs.first; }

class DataSearcher {
public:
  ~DataSearcher() {};

  //// returns a matrix holding <k> closest neighbors of the given label from <labels>
  //// (the label is first calculated from the given image and location)
  //static std::vector<int> FindKClosestLabels(
  //    const ImageCv& matte_gt_cv,
  //    int x,
  //    int y,
  //    int scale, 
  //    int k,
  //    const Matrix& labels) {
  //  // get the label vector from the given image
  //  int patch_size = DataProvider::getPatchSize(scale);
  //  cv::RotatedRect rot_rect(cv::Point2f((x + patch_size)/2, (y + patch_size)/2), cv::Size2f(patch_size, patch_size), 0.f);
  //  ImageCv label_cv = DataProvider::GetLabelVector(matte_gt_cv, rot_rect);
  //  // convert the label to Eigen
  //  Eigen::MatrixXf label;
  //  cv::cv2eigen(label_cv, label);
  //  // find the closest 20
  //  return FindKClosestRows(static_cast<RowVector>(label), labels, k);
  //}

  // finds indices of <k> rows in <matrix> that are closest neighbors of <vector>
  static std::vector<int> FindKClosestRows(const RowVector& vector, const Matrix& matrix, int k) {
    // compute distances from each row in <matrix> to <vector>
    Matrix distances = (matrix.rowwise() - vector).rowwise().squaredNorm();
  
    // add an index to each float value
    std::vector<FloatInt> dist_indices;
    dist_indices.reserve(distances.rows());
    for (int d = 0; d < distances.rows(); ++d) {
      dist_indices.push_back(FloatInt(distances(d), d));
    }
    // find <k> closest rows
    std::partial_sort(dist_indices.begin(), dist_indices.begin() + k, dist_indices.end(), comparator2);

    std::vector<int> k_best_indices;
    k_best_indices.reserve(k);
    for (std::vector<FloatInt>::iterator it = dist_indices.begin();
        it != dist_indices.begin() + k;
        ++it) {
      k_best_indices.push_back(it->second);
    }
    return k_best_indices;
  }

  // returns a sub-patch that's defined but <rot_rect>
  Matrix GetPatch(const Image& big_patch, RotatedRect rot_rect);

private:
  DataSearcher() {};
  DataSearcher(const DataSearcher&) {};
};

#endif