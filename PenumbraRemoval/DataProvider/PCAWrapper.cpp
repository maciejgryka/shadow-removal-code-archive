#include "PCAWrapper.h"
#include <fstream>

#include <DataProvider/DataProvider.h>

PCAWrapper::PCAWrapper(const cv::Mat& data, int n, const std::string& file_name):
    identity_(false) {
	Init(data, n, file_name);
}

PCAWrapper::PCAWrapper(const EigenMat& data, int n, const std::string& file_name):
    identity_(false) {
	cv::Mat data_cv;
  cv::eigen2cv(data, data_cv);
  Init(data_cv, n, file_name);
}

PCAWrapper::PCAWrapper(const std::string& fileName) {
  // check if file exists
	std::ifstream pcaFile(fileName);
	bool fileIsGood = pcaFile.good();
	pcaFile.close();
	assert(fileIsGood);
	
	// read in serialized file
	deserialize(fileName);
}

void PCAWrapper::Init(const cv::Mat& data, int n, const std::string& file_name) {
  if (data.cols == 1) {
		identity_ = true;
		return;
	}
  // if input dimensionality is smaller than output dimensionality n,
  // keep all components
	if (data.cols <= n) { n = 0; }
	// create PCA object from scratch
	pca_ = cv::PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, n);
  // serialize if filename given
	if (!file_name.empty()) { serialize(file_name); }
}

EigenMat PCAWrapper::BackProject(EigenMat vec) const {
	cv::Mat vecCv;
	cv::eigen2cv(vec, vecCv);
	vecCv = BackProject(vecCv);
	DataProvider::CvToEigen(vecCv, vec);
	return vec;
}

EigenMat PCAWrapper::Project(EigenMat vec) const {
	if (identity_) { return vec; }
	cv::Mat vecCv;
	cv::eigen2cv(vec, vecCv);
	vecCv = Project(vecCv);
	DataProvider::CvToEigen(vecCv, vec);
	return vec;
};

void PCAWrapper::deserialize(const std::string& fileName) {
	// fs is closed automatically on destruction
	cv::FileStorage fs(fileName, cv::FileStorage::READ);

	fs["identity"] >> identity_;
	if (identity_) { return; }
	fs["eigenvectors"] >> pca_.eigenvectors;
	fs["eigenvalues"] >> pca_.eigenvalues;
	fs["mean"] >> pca_.mean;
}

void PCAWrapper::serialize(const std::string& fileName) const {
	// fs is closed automatically on destruction
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
	
	fs << "identity" << identity_;
	if (identity_) { return; }

	fs << "eigenvectors" << pca_.eigenvectors;
	fs << "eigenvalues" << pca_.eigenvalues;
	fs << "mean" << pca_.mean;
}