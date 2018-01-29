#ifndef PCA_WRAPPER_H
#define PCA_WRAPPER_H

#include <string>
#include <iostream>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "Types.h"


class PCAWrapper {
public:
	PCAWrapper() {};
	
	// creates PCAWrapper object based on dataMatrix (vectors stored as rows)
	// optional int n is the max number of components (default = 0 keeps all components)
	// optional string fileName allows to specify file in which to store the resulting PCA object
	PCAWrapper(const cv::Mat& data, int n = 0, const std::string& fileName = "");
	PCAWrapper(const EigenMat& dataMatrix, int n = 0, const std::string& fileName = "");
	// read in the object from file
	explicit PCAWrapper(const std::string& fileName);
	explicit PCAWrapper(const cv::PCA& pca): pca_(pca) {};

	~PCAWrapper(void) {};

  cv::Mat BackProject(cv::Mat vec) const {
		if (identity_) { return vec; }
    return pca_.backProject(vec);
	};
	EigenMat BackProject(EigenMat vec) const;

	cv::Mat Project(cv::Mat vec) const {
		if (identity_) { return vec; }
		return pca_.project(vec);
	};
	EigenMat Project(EigenMat vec) const;

  int NOriginalDims() const {
    return pca_.mean.cols;
  }

  int NProjectedDims() const {
    return pca_.eigenvalues.rows;
  }

  cv::Mat eigenvector(int i) const {
    return pca_.eigenvectors.row(i);
  }

	void serialize(const std::string& fileName) const;
	void deserialize(const std::string& fileName);
private:
  void Init(const cv::Mat& data, int n, const std::string& file_name);
	cv::PCA pca_;
	bool identity_;
};

#endif