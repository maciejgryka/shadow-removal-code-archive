#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <string>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "Types.h"

#include "PCAWrapper.h"
//#include "Clusterer.h"
#include "AlignedPatch.h"

#include <PenumbraRemoval/ImageTriple.h>

#define PATH_LENGTH 200
#define IM_TYPE CV_32F
#define MAX_PATCH_SIZE 16

#define PI 3.14159265358979323846
#define PI_180 (PI/180)

#define PATCH_FEATURE_EXTENSION 2
#define PATCH_MARGIN 256

// WARNING: portable?
// typedef for 0..255 integers
typedef unsigned char PixelData;

class DataProvider {
public:
	enum CHANNEL {
		BLUE = 0,
		GREEN,
		RED,
    LIGHTNESS,
		ALL
	};

	enum IM_CAT {
		SHAD = 0,
		NOSHAD,
		MASK,
		MASKP,
    UNSHAD_MASK,
		MATTE,
    GMATTE,
		MATTE_GT,
		UNSHAD
	};

  enum MATRIX_TYPE {
    MATRIX_FEATURE,
    MATRIX_LABEL
  };


  DataProvider() {}

  ~DataProvider() {}

  void set_active_features(const std::map<std::string, bool> activeFeatures) {
    active_features_ = activeFeatures;
  }

	static int getPatchSize(int scaleId) {
		return int(pow(2.0, double(scaleId)));
	}

	static std::string insertScaleNum(std::string s, int scale) {
		char c_str[PATH_LENGTH];
		sprintf(c_str, s.c_str(), scale);
		return std::string(c_str);
	}
  
	static std::string getImageFileName(const std::string& name, int im_cat, const std::string& extension = ".png");
  static std::vector<ImageTriple> getImagePaths(const std::string& imPath, const std::vector<std::string>& imageNames, int nImages = 0);
  static std::vector<std::string> getLinesAsStrVec(const std::string& file);

  // returns a vector of points masked by the mask
  // assumes that given image is a float matrix
  static std::vector<cv::Point> getMaskedPixels(const cv::Mat& mask, int patchSize);
  	
  static cv::Mat getMatte(const ImageTriple& imt, int channel);

	static cv::Mat getShadow(const ImageTriple& imt, int channel) {
		return imreadFloat(imt.getShadow(), channel);
	}

  static cv::Mat getGmatte(const ImageTriple& imt, int channel) {
		return imreadFloat(imt.getGmatte(), channel);
	}

  // get n_labels samples from the data set by sampling each of the given
  // images equally
  // depending on aign_rot and align_trans there might be alignment happening
  void GetLabelSubset(
      const std::vector<ImageTriple>& imtv, 
      int scale,
      CHANNEL channel,
      int n_labels,
      bool align_rot,
      int align_trans,
      EigenMat* labels,
      EigenMat* features);

  // return feature patch from image computed at pos
  cv::Mat FeaturePatchFromImage(
      const cv::Mat& image,
      const cv::Mat& mask,
      const cv::Mat& gmatte,
      bool align_rot,
      int align_trans,
      AlignedPatch* aligned_patch);

  // return label patch from image computed at pos
  cv::Mat LabelPatchFromImage(
      const cv::Mat& image,
      const AlignedPatch& aligned_patch);

  // finds the best alignment of rot_rect within image, aligns it and returns
  // the alignment params
  // searches in rotation if align_rot == true
  // searches in translation in [-align_trans, +align_trans] range
  static void Align(
      const cv::Mat& image,
      bool align_rot,
      int align_trans,
      AlignedPatch* aligned_patch);

  // effectively undoes the above
  // takes a (big window) patch returned from regression and the original alignment params
  // returns a subpatch after unalignment
  static cv::Mat Unalign(
    const cv::Mat& image_patch,
    const RotatedRectOffset& original_offset,
    cv::RotatedRect* unaligned = NULL);

  static cv::Point2f RotatePoint(const cv::Point2f& point, const float& angle, const cv::Point2f& pivot = cv::Point2f(0.f, 0.f));

  static void DrawRotatedRect(const cv::RotatedRect& rot_rect, const cv::Scalar& color, cv::Mat* img);

  static cv::Mat CutOutRotatedRect(const cv::Mat& image, const cv::RotatedRect& rot_rect);

  // Returns a matrix in which each row is a flattened patch centered at one of
  // the points. Assumes that given images are float matrices.
  void ComputeFeaturesAndLabels(
      const cv::Mat& shad,
      const cv::Mat& matte,
      const cv::Mat& mask,
      const cv::Mat& gmatte,
      const std::vector<cv::Point>& points,
      int patch_size,
      bool align_rot,
      int align_trans,
      cv::Mat* features,
      cv::Mat* labels);

  // gets an entire image and cv::Rect for flexibility of computing features from more than the patch
  cv::Mat GetFeatureVector(
      const cv::Mat& image,
      const cv::Mat& mask,
      const cv::Mat& gmatte,
      const AlignedPatch& aligned_patch);

	EigenMat GetFeatureVectorEigen(
      const cv::Mat& image,
      const cv::Mat& mask,
      const cv::Mat& gmatte,
      const AlignedPatch& aligned_patch) {
		cv::Mat featureVector = GetFeatureVector(image, mask, gmatte, aligned_patch);
		// convert to Eigen
		Eigen::RowVectorXf featureVectorEigen;
		cv::cv2eigen(featureVector, featureVectorEigen);
		return featureVectorEigen;
	}
  
  static cv::Mat GetLabelVector(const cv::Mat& image, const AlignedPatch& aligned_patch);

	static bool FileExists(const std::string& path) {
		std::ifstream ifs(path);
		return ifs.good();
	}

  static cv::Mat imreadFloat(const std::string& path, int channel);
  static cv::Mat imreadFloatHSV(const std::string& path, int channel);

  // get gradient of im in X (x = true) or Y (x = false) direction
  static cv::Mat GetGradient(const cv::Mat& im, bool x, bool sobel = true);

  // features
  static cv::Mat GetFeatureIntensity(const cv::Mat& im);
  static cv::Mat GetFeatureGradientXY(const cv::Mat& im);
  static cv::Mat GetFeatureGradientOrientation(const cv::Mat& im);
  static cv::Mat GetFeatureGradientMagnitude(const cv::Mat& im);
  static cv::Mat GetFeaturePlaneNormal(const cv::Mat& patch);
  // Returns a patch that contains distance transform values (normalized to 0-1 
  // range across the whole image) from the mask edge
  static cv::Mat GetFeatureDistanceTransform(const cv::Mat& mask, const cv::Point& pos);
  // Returns one number that indicates the polar angle to the patch when measured
  // from the center of the mask (range 0-2*PI). Center of teh mask is the point
  // with maximum distance transform value.
  static cv::Mat GetFeaturePolarAngle(const cv::Mat& mask, const cv::Point& pos);

  static cv::Mat GetFeatureGmatte(const cv::Mat& gmatte, const AlignedPatch& aligned_patch);

  static cv::Mat DownsamplePatch(const cv::Mat& patch, int times = 1);

	static int CopyIntoVector(const cv::Mat& src, cv::Mat& dst, int pos) {
		int p(0);
		for (; p < src.cols;  ++p) {
			dst.at<float>(0, p+pos) = src.at<float>(0, p);
		}
		return p + pos;
	}

  static void CvToEigen(const cv::Mat& mat_cv, EigenMat& mat_eigen) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> emcm;
    cv::cv2eigen(mat_cv, emcm);
    mat_eigen = EigenMat(emcm);
  }

  //static void EigenToCV(const EigenMat mat_eigen, cv::Mat* mat_cv) {

  //}

	// serialize mat into given fileName as matName
	// the file has to have ".xml" extension
	static void serializeCvMat(const cv::Mat& mat, const std::string& fileName) {
		// fs is closed automatically on destruction
		cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
		fs << "mat" << mat;
	}

	static cv::Mat deserializeCvMat(const std::string& fileName) {
		cv::FileStorage fs(fileName, cv::FileStorage::READ);
		cv::Mat mat;
		fs["mat"] >> mat;
		return mat;
	}

  template <typename T>
  static void serializeEigenMat(const EigenMat& data, const std::string& fileName, bool eight_bit = false) {
    size_t n_rows(data.rows());
    size_t n_cols(data.cols());

    // open the file for binary writing
    std::ofstream opFile;
    opFile.open(fileName, std::ios::out | std::ios::binary);
    
    // write two integers for number of rows and columns
    opFile.write((char*)(&n_rows), sizeof(size_t));
    //opFile << rows << " ";
    opFile.write((char*)(&n_cols), sizeof(size_t));
    //opFile << cols << " ";
  
    T temp;
	  for (int rIt(0); rIt < n_rows; ++rIt) {
		  for (int cIt(0); cIt < n_cols; ++cIt) {
        if (eight_bit) {
          if (data(rIt, cIt) < 0.0f) { 
            temp = 0; 
          } else if (data(rIt, cIt) > 0.996f) { 
            temp = 255;
          } else {
            temp = static_cast<T>(data(rIt, cIt)*255.f);
          }
        } else {
          temp = data(rIt, cIt);
        }
        opFile.write((char*)(&temp), sizeof(T));
        //opFile << temp << " ";
        //if (data(rIt, cIt) > 0.99f && static_cast<int>(temp) < 250) {
        //  std::cerr << "bug: " << data(rIt, cIt) << "f == " << static_cast<int>(temp) << "char" << std::endl;
        //}
		  }
	  }
	  opFile.close();
  }

  template <typename T>
  static EigenMat deserializeEigenMat(const std::string& fileName, bool eight_bit = false) {
    size_t n_rows;
    size_t n_cols;
    
    std::ifstream inFile;
	  inFile.open(fileName, std::ios::in | std::ios::binary);
     
    // read in number of rows and columns
    inFile.read((char*)(&n_rows), sizeof(size_t));
    //inFile >> rows;
    inFile.read((char*)(&n_cols), sizeof(size_t));
    //inFile >> cols;
		
	  EigenMat data(n_rows, n_cols);
    T temp;
    // read the matrix
    for (int r(0); r < n_rows; ++r) {
      for (int c(0); c < n_cols; ++c) {
        inFile.read((char*)(&temp), sizeof(T));
        //inFile >> temp;
        data(r, c) = static_cast<float>(temp);
        if (eight_bit) {
          data(r, c) /= 255.f;
        }
      }
    }

    inFile.close();
	  return data;
  }

  // writes the matrix to disk as a CSV file (by default first line contains
  // two integers that represent matrix dimensionality)
  static void serializeEigenMatAscii(const EigenMat& data, const std::string& fileName, bool write_dimensions = true);
  static EigenMat deserializeEigenMatAscii(const std::string& fileName);
  
  // increases the size of the given rect by a multiplier withour changing the angle or center
  static void ExpandRotRect(const float& multiplier, cv::RotatedRect* rot_rect) {
    if (multiplier == 1) { 
      return;
    }
    rot_rect->size.width *= multiplier;
    rot_rect->size.height *= multiplier;
  }

  static void ShrinkRotRect(const float& multiplier, cv::RotatedRect* rot_rect) {
    ExpandRotRect(1.f/multiplier, rot_rect);
  }

  static bool IsInImage(const cv::Rect& r, const cv::Mat& image) {
    cv::Point max_point(r.x + r.width, r.y + r.height);
    return r.x > 0 && r.y > 0 && max_point.x < image.cols && max_point.y < image.rows;
  }

private:
  std::deque<bool> GetPointSelection(int nPoints, float inclusion_chance);
  std::vector<cv::Point> getSelectedPoints(
      const std::vector<cv::Point>& points,
      const std::deque<bool>& selection);

  static void Downsample(cv::Mat* patch, int interpolation = 1);
  std::map<std::string, bool> active_features_;

  //cv::Mat distance_transform
};

#endif