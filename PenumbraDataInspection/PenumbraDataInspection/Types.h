#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//static const int CV_IMAGE_TYPE =

typedef Eigen::MatrixXf Image;
typedef cv::Mat ImageCv;

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;

typedef std::pair<float, int> FloatInt;

#endif TYPES_H