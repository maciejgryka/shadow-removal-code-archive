#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

#include <Eigen/Core>

#include "RandomForestCommon.h"

/*	
    Normal distribution with online updates following
    http://www.johndcook.com/standard_deviation.html 
*/
class NormalDistribution {
public:
  NormalDistribution(EigenMat points): n_points_(0)	{ Update(points); }
	NormalDistribution(): n_points_(0) {};
	NormalDistribution(Eigen::VectorXf mean, EigenMat covariance): 
		  mean_(mean), 
		  covariance_(covariance), 
		  n_points_(1) {};
	
	~NormalDistribution() {};

	void Update(const EigenMat& newPoints) {
		int k = 0;
		if (!n_points_) {
			mean_ = newPoints.row(k);
			covariance_ = EigenMat(mean_.cols(), mean_.cols());
			covariance_.setZero();
			++n_points_;
			++k;
		}

		Eigen::RowVectorXf diff_old;
		// compute the mean and covariance
		for (; k < newPoints.rows(); ++k) {
			++n_points_;
			diff_old = newPoints.row(k) - mean_;		
			mean_ += diff_old / static_cast<float>(n_points_);
			covariance_ += diff_old.transpose() * (newPoints.row(k) - mean_);
		}
	};

	Eigen::RowVectorXf mean() const { return mean_; };
	EigenMat covariance() const { return covariance_ / static_cast<float>(n_points_ - 1); };

private:
	Eigen::RowVectorXf mean_;
	EigenMat covariance_;
	int n_points_;	// number of points so far
};

#endif