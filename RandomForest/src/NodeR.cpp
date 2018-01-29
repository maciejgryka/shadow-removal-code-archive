#include "NodeR.h"

#include <iostream>
#include <fstream>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <DataProvider/Serializer.h>

using std::vector;
using namespace RandomForest;

NodeR::NodeR(
    NodeIdType id,
    const EigenMatCM& data,
    const EigenMat& labels,
    const vector<int>& samples):

	  id_(id),
    n_samples_(samples.size()),
    samples_(samples),
    nd_(SubsetRowwise(labels, samples)) {
	InitNode(data, labels);
}

NodeR::NodeR(
    NodeIdType id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity):

	  id_(id),
	  is_leaf_(is_leaf),
	  split_dim_(split_dim),
	  split_thresh_(split_thresh),
    impurity_(impurity),
    left_(NULL),
    right_(NULL),
    parent_(NULL) {
  assert(!is_leaf);
}

NodeR::NodeR(
    NodeIdType id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity,
    const NormalDistribution& nd,
    const vector<int>& samples):

	  id_(id),
	  is_leaf_(is_leaf),
	  split_dim_(split_dim),
	  split_thresh_(split_thresh),
    impurity_(impurity),
	  nd_(nd),
    n_samples_(samples.size()),
	  samples_(samples),
    left_(NULL),
    right_(NULL),
    parent_(NULL) {
	assert(is_leaf);
}

NodeR::~NodeR(void) {	
	if (!is_leaf_) {
		delete left_;
		delete right_;
	}
}

void NodeR::InitNode(const EigenMatCM& data, const EigenMat& labels) {
	CalcDimLimits(data);

	impurity_ = DistributionEntropy(nd_);
	is_leaf_ = false;
}

void NodeR::CalcDimLimits(const EigenMatCM& data) {
	assert(samples_.size() > 0);

	dim_mins_ = vector<double>(data.cols());
	dim_maxes_ = vector<double>(data.cols());

	for (int col = 0; col < data.cols(); col++) {
		dim_mins_[col] = DBL_MAX;
		dim_maxes_[col] = DBL_MIN;
	}

	for (int col = 0; col < data.cols(); ++col) {
    dim_mins_[col] = data.col(col).minCoeff();
    dim_maxes_[col] = data.col(col).maxCoeff();
	}
}

NormalDistribution NodeR::CalcFeatureDist(const EigenMat& data) {
	return getSampleDistr(data, samples_);
}

NodeR* NodeR::GetTerminalNode(const RowVector& features) {
  if (is_leaf_) {
    return this;
  }
  if (features(split_dim_) < split_thresh_) {
    if (left_ == NULL) {
      std::cerr << "ERROR: node " << id_ << " is_leaf = " << is_leaf_ << " and has no left child." << std::endl;
      return this;
    }
  } else {
    if (right_ == NULL) {
      std::cerr << "ERROR: node " << id_ << " is_leaf = " << is_leaf_ << " and has no right child." << std::endl;
      return this;
    }
  }
  return features(split_dim_) < split_thresh_ ? left_->GetTerminalNode(features) : right_->GetTerminalNode(features);
}

NormalDistribution NodeR::EvaluatePoint(const RowVector& features) {
  return GetTerminalNode(features)->nd_;
}

float NodeR::GetFeatureDistanceMahalanobis(
    const RowVector& featurePoint,
    const EigenMat& data) {
  NormalDistribution featureDistribution = getSampleDistr(data, samples_);
  RowVector difference = featurePoint - featureDistribution.mean();
  EigenMat diagonalCov = featureDistribution.covariance().diagonal().asDiagonal();

  float dist = (difference * diagonalCov.inverse() * difference.transpose())[0];

  return dist;
}

void NodeR::WriteNode(std::ofstream& file) const {
	file << id_ << std::endl;
	file << is_leaf_ << std::endl;
  file << n_samples_ << std::endl;
	file << split_dim_ << std::endl;
	file << split_thresh_ << std::endl;
  file << impurity_ << std::endl;
	
	if (is_leaf_) {
    //// DBG
    //if (id_ == 31) {
    //  std::cout << std::endl;
    //  std::cout << "saving node " << id_ << ":" << std::endl;
    //  std::cout << std::endl << nd_.mean() << std::endl;
    //  std::cout << std::endl << nd_.covariance() << std::endl;
    //  std::string s;
    //  std::getline(std::cin, s);
    //}
    file << matrixString(nd_.mean()) << std::endl;
	  file << matrixString(nd_.covariance()) << std::endl;
    file << intVectorString(samples_) << std::endl;
	} else {
    //file << RandomForest::StdVectorToString(info_gains_) << std::endl;
		left_->WriteNode(file);
		right_->WriteNode(file);
	}
}

void NodeR::WriteNodeBin(std::ofstream& out_stream) const {
  StreamWriter sw(&out_stream);
  sw.WriteStream<NodeIdType>(id_, out_stream);
  sw.WriteInt(static_cast<int>(is_leaf_));
  sw.WriteInt(n_samples_);
  sw.WriteInt(split_dim_);
  sw.WriteFloat(split_thresh_);
  sw.WriteFloat(impurity_);

  if (is_leaf_) {
    sw.WriteFloatMatrix(nd_.mean());
    sw.WriteFloatMatrix(nd_.covariance());
    sw.WriteIntVector(samples_);
  } else {
    //sw.WriteFloatVector(info_gains_);
    left_->WriteNodeBin(out_stream);
    right_->WriteNodeBin(out_stream);
  }
}

void NodeR::ReadNodeBin(std::ifstream& in_stream) {
  StreamReader sr(&in_stream);

  id_ = sr.ReadStream<NodeIdType>(in_stream);
  is_leaf_ = sr.ReadInt() == 1;
  n_samples_ = sr.ReadInt();
  split_dim_ = sr.ReadInt();
  split_thresh_ = sr.ReadFloat();
  impurity_ = sr.ReadFloat();

  //std::cout << "reading node " << id_ << std::endl;
  //std::cout << "is_leaf " << is_leaf_ << std::endl;
  //std::cout << "n_samples " << n_samples_ << std::endl;

  Eigen::VectorXf temp_mean;
  EigenMat temp_cov;
  if (is_leaf_) {
    temp_mean = sr.ReadFloatMatrix().row(0);
    temp_cov = sr.ReadFloatMatrix();
    nd_ = NormalDistribution(temp_mean, temp_cov);
    samples_ = sr.ReadIntVector();
  }
  //else {
    //info_gains_ = sr.ReadFloatVector();
  //}
}

std::string NodeR::matrixString(const EigenMat& mat) {
	std::ostringstream oss;
	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			oss << mat(r, c);
			if ((r+1)*(c+1) < mat.rows() * mat.cols()) {
				oss << ",";
			}
		}
	}
	return oss.str();
}

std::string NodeR::intVectorString(const vector<int> vec) {
  std::ostringstream oss;
  oss << vec[0];
  for (int el = 1; el < int(vec.size()); ++el) {
    oss << "," << vec[el];
  }
  return oss.str();
}

//NormalDistribution RandomForest::GetProjectedSampleDistr(
//    const EigenMat& data,
//    const PCAWrapper& pcaw,
//    const vector<int>& samples) {
//  int n_dim_out = pcaw.NProjectedDims();
//  size_t n_samples = samples.size();
//
//	EigenMat l(n_samples, n_dim_out);
//	for (int r = 0; r < l.rows(); r++) {
//    l.row(r) = pcaw.Project(data.row(samples[r]));
//	}
//	return NormalDistribution(l);
//}

NormalDistribution RandomForest::getSampleDistr(
    const EigenMat& data,
    const vector<int>& samples) {
	int nDimOut = data.cols();
	int nSamples(samples.size());

	EigenMat l(int(samples.size()), data.cols());
	for (int r = 0; r < l.rows(); r++) {
    l.row(r) = data.row(samples[r]);
	}
	return NormalDistribution(l);
}

bool IsFiniteNumber(double x) {
  return (x <= DBL_MAX && x >= -DBL_MAX); 
} 

inline double RandomForest::DistributionEntropy(const NormalDistribution& nd) {
  double det = nd.covariance().determinant();
  // in case det < 0 something is screwed up - most likely we don't have enough
  // samples to calculate covariance properly, or there's a linear dependence
  // between them - return large impurity to indicate an invalid split
  if (det < 0.0) {
    std::cerr << "Invalid det(cov) : " << det << std::endl;
    return DBL_MAX;
  }
  // if the determinant is 0, this is a perfect case - all samples are the same!
  // that means we have to return perfect impurity score, but what does that mean?
  // we set det to DBL_MIN because we can't compute log of 0
  if (det == 0.0) { det = DBL_MIN; }
  // compute log of the determinant
  double det_log = log(det);
  // sanity check - should never happen!
  if (!IsFiniteNumber(det_log)) {
    std::cerr << "Invalid log(det(cov)) : " << det_log << std::endl;
    return DBL_MAX;
  } 
  return det_log;
}

// construct a matrix as a subset of another
inline EigenMat RandomForest::SubsetRowwise(
    const EigenMat& superset,
    const vector<int>& row_indices) {
  size_t n_rows(row_indices.size());
  EigenMat subset(n_rows, superset.cols());
	for (size_t i(0); i < n_rows; ++i) {
    subset.row(i) = superset.row(row_indices[i]);
	}
  return subset;
}

double RandomForest::Entropy(const EigenMat& labels, const std::vector<int>& samples) {
	return DistributionEntropy(SubsetRowwise(labels, samples));
}