#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataSearcher.h"

//using namespace cv;

typedef std::vector<std::string> StringVector;
typedef std::vector<cv::Rect> RectVector ;

typedef std::vector<RectVector> PatchVector;

const std::string main_dir("G:\\projects\\penumbraRemoval\\2012-06-29\\");
const std::string data_dir(main_dir + "output\\size1861\\data\\");
const int scale = 6;
const int psize = DataProvider::getPatchSize(scale);
const int k = 20;

StringVector image_names;
PatchVector patches;

void PopulateNames(StringVector* image_names, PatchVector* patches) {
  image_names->push_back("003_0037_225_concrete3");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(128, 192, psize, psize));
  patches->back().push_back(cv::Rect(192, 256, psize, psize));
  patches->back().push_back(cv::Rect(320, 320, psize, psize));

  image_names->push_back("015_0033_6_soil9");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(192, 192, psize, psize));
  patches->back().push_back(cv::Rect(192, 256, psize, psize));
  patches->back().push_back(cv::Rect(256, 320, psize, psize));

  image_names->push_back("024_0057_7_concrete11");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(256, 128, psize, psize));
  patches->back().push_back(cv::Rect(192, 320, psize, psize));

  image_names->push_back("125_0016_154_brick7");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(256, 192, psize, psize));
  patches->back().push_back(cv::Rect(256, 64, psize, psize));
  patches->back().push_back(cv::Rect(256, 320, psize, psize));

  image_names->push_back("125_0016_266_wall2");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(128, 128, psize, psize));
  patches->back().push_back(cv::Rect(128, 384, psize, psize));
  patches->back().push_back(cv::Rect(192, 128, psize, psize));

  image_names->push_back("125_0018_281_tile5");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(320, 192, psize, psize));
  patches->back().push_back(cv::Rect(320, 256, psize, psize));
  patches->back().push_back(cv::Rect(256, 256, psize, psize));

  image_names->push_back("125_0037_17_wall13");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(128, 256, psize, psize));
  patches->back().push_back(cv::Rect(192, 256, psize, psize));
  patches->back().push_back(cv::Rect(64, 256, psize, psize));

  image_names->push_back("136_0015_160_soil7");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(128, 384, psize, psize));
  patches->back().push_back(cv::Rect(128, 320, psize, psize));
  patches->back().push_back(cv::Rect(192, 192, psize, psize));

  image_names->push_back("244_0075_2_tile11");
  patches->push_back(RectVector());
  patches->back().push_back(cv::Rect(256, 128, psize, psize));
  patches->back().push_back(cv::Rect(256, 192, psize, psize));
  patches->back().push_back(cv::Rect(256, 256, psize, psize));
}

std::string IntToString(int val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

int StringToInt(const std::string& val) {
  return atoi(val.c_str());
}

const int n_mean = 8;

int main(int argc, char* argv[]) {
  PopulateNames(&image_names, &patches);
  //std::string features_path = data_dir + "features6.dat";
  std::string labels_path = data_dir + "labels6.dat";;
  Matrix labels = DataProvider::deserializeMatrixXf(labels_path);

  int n_images = static_cast<int>(image_names.size());
  for (int i = 0; i < n_images; ++i) {
    // read the matte image
    std::string matte_gt_path = main_dir + "test_images\\" + image_names[i] + "_matte.png";
    ImageCv matte_gt_cv = cv::imread(matte_gt_path);
    // extract the red channel
    std::vector<ImageCv> matte_gt_ch(3);
    cv::split(matte_gt_cv, matte_gt_ch);
    matte_gt_cv = matte_gt_ch[0];
    matte_gt_cv.convertTo(matte_gt_cv, CV_32F, 1.f/255.f);

    // for each specified patch in the image
    int n_patches = static_cast<int>(patches[i].size());
    for (int p = 0; p < n_patches; ++p) {
      // get the target rect and write image to disk
      cv::Rect r = patches[i][p];
      ImageCv label_cv = DataProvider::getLabelVector(matte_gt_cv, r);
      Image label;
      cv::cv2eigen(label_cv, label);
      label.resize(psize, psize);
      cv::eigen2cv(label, label_cv);
      std::string out_dir(data_dir + image_names[i] + "\\patch_" + IntToString(r.x) + "_" + IntToString(r.y) + "\\");
      system(std::string("mkdir " + out_dir).c_str());
      cv::imwrite(out_dir + "label_target.png", label_cv*255.f);

      // find k closest labels
      std::vector<int> closest_labels = DataSearcher::FindKClosestLabels(matte_gt_cv, r.x, r.y, scale, k, labels);
      // extract them and write to disk
      ImageCv mean(label_cv);
      mean.setTo(0.f);
      for (int l = 0; l < k; ++l) {
        Image im = labels.row(closest_labels[l]);
        im.resize(psize, psize);
        ImageCv im_cv;
        cv::eigen2cv(im, im_cv);
        std::stringstream ss;
        ss << l;
        cv::imwrite(out_dir  + "\\label" + ss.str() + ".png", im_cv*255.f);
        // if we're within the first n_mean images, add this to the mean
        if (l < n_mean) {
          mean += im_cv;
        }
      }
      mean /= n_mean;
      cv::imwrite(out_dir + "label_mean.png", mean*255.f);
    }
  }
}
