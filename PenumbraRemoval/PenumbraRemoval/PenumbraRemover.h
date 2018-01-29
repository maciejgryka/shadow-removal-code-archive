#ifndef PENUMBRA_REMOVER_H
#define PENUMBRA_REMOVER_H

#include <string>
#include <list>
#include <map>

#include <DataProvider/PCAWrapper.h>
#include <DataProvider/Types.h>
#include <Ensemble.h>
#include <EnsembleParams.h>
#include <Regularization/RegGraph.h>
#include "Options.h"


#ifndef __OPENCV_CORE_HPP__
namespace cv {
class Mat;
class Rect;
};
#endif

#ifndef EIGEN_CORE_H
namespace Eigen {
class MatrixXf;
};
#endif


//#define N_TREES 100
// number of constant patches at the finest regularization scale
#define N_CONSTANT_PATCHES 32

class PenumbraRemover {
public:
  typedef std::pair<float, int> FloatInt;

	PenumbraRemover(int finestScale, int nScales, int scaleStep, float unary_cost_scaling);

	~PenumbraRemover(void);

	cv::Mat Test(
      const cv::Mat& shad_im,
	    const cv::Mat& mask_im,
	    const cv::Mat& gmatte_im,
	    const cv::Mat& unshad_mask_im,
      const Options& options,

	void TrainScale(
      int scale_id,
	    const EigenMat& data,
	    const EigenMat& labels,
	    const Options& options,
	    const std::string& ensemble_file = "",
	    const std::string& pcaw_file = "",
      bool compute_pca = true,
      const std::string& leaves_path = "");

  void TestProjection(
      int scale_id,
	    const EigenMat& labels,
	    const std::string& pca_file,
	    const std::string& out_folder);

  // sets al pixels within margin of boundaty to zero
  void SetMarginsTo(int margin, float val, cv::Mat* img);

	//void WriteTrainingSetOutput(
 //     int scaleId,
	//    const EigenMat& data,
	//    const std::string& ensembleFile,
	//    const std::string& pcaFile,
	//    const std::string& outFile);

	static bool IsAllZeros(const cv::Mat& im) {
    return cv::countNonZero(im) == 0;
  }

  //void set_ensemble_params(const EnsembleParams& ensemble_params, int scale) {
  //  ensemble_params_[scale] = ensemble_params;
  //};
  void set_active_features(const std::map<std::string, bool>& active_features) {
    active_features_ = active_features;
  };
  void set_data_file(const std::string& data_file) {
    data_file_ = data_file;
  };
  void set_labels(int scale, EigenMat* labels) {
    labels_matrix_[scale] = labels;
  };
  void set_unary_cost(int unary_cost) {
    unary_cost_ = unary_cost;
  };
  void set_uniform_finest(bool uniform_finest) {
    uniform_finest_ = uniform_finest;
  };

private:
  static const double kCharToFloatCoeff;// = 1.0/255.0;
  static const double kFloatToCharCoeff;// = 255.0;

  // returns vector of Rects, each representing a patch at the given scale
	std::list<cv::Rect> GetPatchCoords(const cv::Mat& im, int scale);
	// removes from the provided list patches that do not include any masked
  // pixels and fills in the passed vector with 0's and 1's depending on
  // whether a given patch is masked
	void RemoveUnwantedPatches(
      std::list<cv::Rect>& patchCoords,
      const cv::Mat& maskIm,
      std::vector<int>* maskedPatches);
  // Returns the difference in nDCG measure at p between ideally sorted
  // elements (dists_gt) and elements sorted according to the unary cost.
  // dists_gt hold <distance_from_gt, index>
  // unary_costs_sorted hold <unary_cost, index>
  // and both are sorted in ascending order
  float GetNdcg(
      const std::vector<FloatInt>& dists_gt,
      const std::vector<FloatInt>& unary_costs_sorted,
      int p);

  // create a guessed matte from shadow image and mask
  cv::Mat GuessMatte(
      const cv::Mat& shad,
      const cv::Mat& mask,
      const cv::Mat& unshad_mask,
      const Options& options);

  void GetNonZeroPoints(const cv::Mat& img, std::vector<cv::Point2i>* non_zero_points);

  // keeps the first n labels with smallest corresponding unary costs
  // and throws away the rest
  void KeepTopNLabels(
      int n,
      std::vector<EigenMat>* labels,
      std::vector<float>* unary_costs);

  std::map<int, bool> is_trained_;
	int finest_scale_;
	int  n_scales_;
  int scale_step_;
  float unary_cost_scaling_;
  bool uniform_finest_;
  std::map<std::string, bool> active_features_;
  std::string data_file_;
	//std::map<int, EnsembleParams> ensemble_params_;

	std::map<int, PCAWrapper> pcaws_;
	std::map<int, RandomForest::Ensemble> forests_;

  int unary_cost_; // 0 - uniform,
                   // 1 - Mahalanobis distance in feature space
                   // 2 - distance to mean across trees

  std::map<int, EigenMat*> labels_matrix_;

  // not allowed
	PenumbraRemover();
};

#endif