#include "PlaneFitter.h"

#include <iostream>

float round(float f) {
	return (f > 0.0) ? floor(f + 0.5f) : ceil(f - 0.5f);
}

Eigen::Vector4f PlaneFitter::GetBestPlaneRansac(
    const EigenMat points,
    int n_iterations,
    float threshold) {
  int max_inliers(0);
  float min_distance(FLT_MAX);
  Eigen::Vector4f best_plane(0.f, 0.f, 0.f, 0.f);

  for (int i(0); i < n_iterations; ++i) {
    int iter_inliers(0); // number of inliers at this iteration
  	float iter_distance(0.f); // sum of distances for all inliers
    
    Eigen::Vector4f plane(GetRandomPlane(points));
    float normalization_factor(sqrt(plane.x()*plane.x() + plane.y()*plane.y() + plane.z()*plane.z()));
    
    for (int p(0); p < points.rows(); ++p) {
      float distance(GetPlanePointDistanceUnnormalized(plane, points.row(p))/normalization_factor);
			if (distance < threshold) {
				++iter_inliers;
				iter_distance += distance;
			}
		}

    if (iter_inliers > max_inliers || (iter_inliers == max_inliers && iter_distance < min_distance)) {
      best_plane = plane;
			max_inliers = iter_inliers;
			min_distance = iter_distance;
		}
  }
  std::cout << max_inliers << std::endl;
  return best_plane;
}

//Eigen::Vector4f PlaneFitter::GetBestPlaneLsq(const EigenMat& plane) {
//  // using least squres
//  Mat plane_points(non_zero_nm_points.size(), 3+1, CV_32F);
//  Mat temp_row(1, 3+1, CV_32F);
//  temp_row.at<float>(3) = 1.0f;
//  int row_index = 0;
//  for (auto point = non_zero_nm_points.begin();
//      point != non_zero_nm_points.end();
//      ++point) {
//    // also pad the last column of the point matrix with zeros
//    temp_row.at<float>(0) = static_cast<float>(point->x);
//    temp_row.at<float>(1) = static_cast<float>(point->y);
//    temp_row.at<float>(2) = static_cast<float>(fit_to.at<unsigned char>(*point));
//    temp_row.copyTo(plane_points.row(row_index));
//    ++row_index;
//  }
//  cv::SVD svd;
//  Mat w, u, vt;
//  svd.compute(plane_points, w, u, vt);
//
//  //std::cout << vt << std::endl;
//
//  coeffs = vt.row(3);
//  for (vector<Point2i>::const_iterator point = non_zero_m_points.begin();
//      point != non_zero_m_points.end();
//      ++point) {
//    float val = (coeffs(0)*point->x  + coeffs(1)*point->y + coeffs(3)) / -coeffs(2);
//    val /= 255.f; // convert to 0-1 range
//    noshad_blur.at<float>(*point) = val;
//  }
//  break;
//}

Eigen::Vector4f PlaneFitter::GetRandomPlane(const EigenMat& points) {
  // get a matrix with 3 points randomly chosen from the given matrix
  Eigen::Vector3f p1(points.row(GetRandomRowIndex(points.rows())));
  Eigen::Vector3f p2(points.row(GetRandomRowIndex(points.rows())));
  Eigen::Vector3f p3(points.row(GetRandomRowIndex(points.rows())));

  Eigen::Vector4f plane;

	// copied from http://paulbourke.net/geometry/planeeq/
	plane.x() = p1.y()*(p2.z() - p3.z()) + 
              p2.y()*(p3.z() - p1.z()) + 
              p3.y()*(p1.z() - p2.z());
	
  plane.y() = p1.z()*(p2.x() - p3.x()) + 
              p2.z()*(p3.x() - p1.x()) + 
              p3.z()*(p1.x() - p2.x());
	
  plane.z() = p1.x()*(p2.y() - p3.y()) + 
              p2.x()*(p3.y() - p1.y()) + 
              p3.x()*(p1.y() - p2.y());
	
  plane.w() = -(p1.x()*(p2.y()*p3.z() - p3.y()*p2.z()) + 
                p2.x()*(p3.y()*p1.z() - p1.y()*p3.z()) + 
                p3.x()*(p1.y()*p2.z() - p2.y()*p1.z()));

	return plane;
}

bool PlaneFitter::IsValidMask(const EigenMat& mask) {
  return mask.rows() > 0 && mask.cols() > 0;
}
