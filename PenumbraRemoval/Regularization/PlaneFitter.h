#ifndef PLANE_FITTER_H
#define PLANE_FITTER_H

#include <stdlib.h>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <DataProvider/Types.h>

typedef Eigen::Vector2f NormalType;

class PlaneFitter {
public:
	~PlaneFitter() {};

  static Eigen::Vector4f PlaneFitter::GetBestPlaneRansac(
    const EigenMat points,
    int n_iterations,
    float threshold);

  //// takes a square matrix (image patch) and returns plane parameters
  //static Eigen::Vector4f PlaneFitter::GetBestPlaneLsq(
  //    const EigenMat& plane);


private:
  static Eigen::Vector4f GetRandomPlane(const EigenMat& points);
	
  static float GetPlanePointDistanceUnnormalized(
      const Eigen::Vector4f& plane,
      const Eigen::Vector3f& point) {
    return abs(plane.x()*point.x() + plane.y()*point.y() + plane.z()*point.z() + plane.w());
  }

  static int GetRandomRowIndex(const int& n_rows) {
    return static_cast<int>((n_rows-1) * static_cast<float>(rand())/static_cast<float>(RAND_MAX));
  }

  static bool IsValidMask(const EigenMat& mask);

	PlaneFitter();
	PlaneFitter(const PlaneFitter&);
};

#endif // PLANE_FITTER_H