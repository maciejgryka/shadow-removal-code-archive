#include <time.h>
#include <algorithm>

#include <Regularization/PlaneFitter.h>

#include "DataProvider.h"

using namespace std;
using namespace Eigen;
using namespace cv;

std::deque<bool> DataProvider::GetPointSelection(int nPoints, float inclusion_chance) {
  srand(20120408);
  // otherwise assign selection based on inclusionChance
  std::deque<bool> point_selection = std::deque<bool>(nPoints);
  for (int p = 0; p < nPoints; ++p) {
    if (static_cast<float>(rand())/RAND_MAX < inclusion_chance) {
      point_selection[p] = true;
    } else {
      point_selection[p] = false;
    }
  }
  return point_selection;
};

// TODO: inefficient - use pointers to modify in-place
std::vector<cv::Point2i> DataProvider::getSelectedPoints(
    const std::vector<cv::Point2i>& points,
    const std::deque<bool>& selection) {
  assert(points.size() == selection.size());
  std::vector<cv::Point2i> selectedPoints;
  for (int p = 0; p < points.size(); ++p) {
    if (selection[p]) {
      selectedPoints.push_back(points[p]);
    }
  }
  return selectedPoints;
};

string DataProvider::getImageFileName(const string& name, int im_cat, const string& extension) {
	string suffix("");
	switch (im_cat) {
	case SHAD:
		suffix = "_shad";
		break;
	case NOSHAD:
		suffix = "_noshad";
		break;
	case MASK:
		suffix = "_smask";
		break;
  case MASKP:
		suffix = "_maskp";
		break;
  case UNSHAD_MASK:
    suffix = "_pmask";
    break;
	case MATTE:
		suffix = "_matte";
		break;
  case GMATTE:
		suffix = "_gmatte";
		break;
  case MATTE_GT:
		suffix = "_matte_gt";
		break;
	case UNSHAD:
		suffix = "_unshad";
		break;
	}  
	return name + suffix + extension;
};

vector<ImageTriple> DataProvider::getImagePaths(const string& imPath, const vector<string>& imageNames, int nImages) {
	nImages = (nImages == 0 ? int(imageNames.size()) : nImages);
		
	vector<ImageTriple> imtv(nImages);
	for (int i = 0; i < nImages; ++i) {
		imtv[i] = ImageTriple(
        imPath,
        getImageFileName(imageNames[i], SHAD),
        getImageFileName(imageNames[i], NOSHAD),
        getImageFileName(imageNames[i], MASK),
        getImageFileName(imageNames[i], MASKP),
        getImageFileName(imageNames[i], GMATTE));
	}

	return imtv;
}

vector<string> DataProvider::getLinesAsStrVec(const string& file) {
  vector<string> lines;

  string ipFile(file);
  ifstream  infile;
  infile.open(ipFile, ios::in);
  string line;
  while (!getline(infile, line).eof()) {
	  lines.push_back(line);
  }
  infile.close();

  return lines;
}

vector<Point> DataProvider::getMaskedPixels(const Mat& mask, int patchSize) {
	vector<Point> points;
	for (int row = patchSize; row < mask.rows - patchSize; ++row) {
		for (int col = patchSize; col < mask.cols - patchSize; ++col) {
			if (mask.at<float>(row, col) > 0.0f) {
				points.push_back(Point(col, row));
			}
		}
	}
  return points;
}

cv::Mat DataProvider::getMatte(const ImageTriple& imt, int channel) {
	cv::Mat shadow(imreadFloat(imt.getShadow(), channel));
	cv::Mat noShadow(imreadFloat(imt.getNoShadow(), channel));

	// obtain shadow matte
	cv::Mat matte;
	divide(shadow, noShadow, matte);

	return matte;
}

// DBG: define custom random generator to make sure that random_shuffle below is always doing the same thing
// additionally rand is seeded at the beginning of main()
// random generator function:
ptrdiff_t MyRandom(ptrdiff_t i) { return rand()%i; }
// pointer object to it:
ptrdiff_t (*p_myrandom)(ptrdiff_t) = MyRandom;

void DataProvider::GetLabelSubset(
    const std::vector<ImageTriple>& imtv,
    int scale,
    CHANNEL channel,
    int n_labels,
    bool align_rot,
    int align_trans,
    EigenMat* labels,
    EigenMat* features) {
  int rowCount(0);
  EigenMat prevLabelsMat, tempLabelsMat;
  EigenMat prevFeaturesMat, tempFeaturesMat;

  Mat tempLabelsMatCv;
  Mat tempFeaturesMatCv;
  bool isMatInitialized(false);
  int patch_size(getPatchSize(scale));
  int labels_per_image(n_labels / imtv.size());
  // if we take all 3 channels, than each single-channel image should only give
  // a third of labels_per_image
  if (channel == ALL) { labels_per_image /= 3; }
  // HACK
  labels_per_image = labels_per_image == 0 ? 1 : labels_per_image;
  for (int i = 0; i < imtv.size(); ++i) {
    for (int ch = 0; ch < 3; ++ch) {
      if (channel != ALL && channel != ch) {
        continue;
      }
      std::cout << "image " << imtv[i].getShadow() << std::endl;
      Mat mask(imreadFloat(imtv[i].getMask(), ch));
      Mat maskp(imreadFloat(imtv[i].getMaskP(), ch));

      // get appropriate images from which to extract data
      Mat shad(getShadow(imtv[i], ch));
      Mat matte(getMatte(imtv[i], ch));
    
      Mat gmatte;
      if (active_features_["feature_gmatte"]) {
        gmatte = getGmatte(imtv[i], ch);
      }
      // expand images in case we need to compute features from area larger than label patches
      int margin = patch_size * PATCH_FEATURE_EXTENSION;
      copyMakeBorder(shad, shad, margin, margin, margin, margin, BORDER_REFLECT);
      copyMakeBorder(matte, matte, margin, margin, margin, margin, BORDER_REFLECT);
      copyMakeBorder(mask, mask, margin, margin, margin, margin, BORDER_CONSTANT, 0.0f);
      copyMakeBorder(maskp, maskp, margin, margin, margin, margin, BORDER_CONSTANT, 0.0f);
      if (active_features_["feature_gmatte"]) {
        copyMakeBorder(gmatte, gmatte, margin, margin, margin, margin, BORDER_REFLECT);
      }
  
      // get a vector of positions from which we can extract patches
      // half from the whole mask
      vector<Point> valid_points(getMaskedPixels(mask, patch_size));
      // and half from the panumbra mask
      vector<Point> valid_points_p(getMaskedPixels(maskp, patch_size));
      // randomize the point order
      random_shuffle(valid_points.begin(), valid_points.end(), p_myrandom);
      random_shuffle(valid_points_p.begin(), valid_points_p.end(), p_myrandom);
      // keep only first labels_per_image labels
      if (valid_points.size() + valid_points_p.size() > labels_per_image) {
        valid_points.erase(valid_points.begin()+labels_per_image/2, valid_points.end());
        valid_points_p.erase(valid_points_p.begin()+labels_per_image/2, valid_points_p.end());
      }
      // marge the two vectors and reshuffle
      valid_points.insert(valid_points.end(), valid_points_p.begin(), valid_points_p.end());
      random_shuffle(valid_points.begin(), valid_points.end(), p_myrandom);

      // compute features and labels
      ComputeFeaturesAndLabels(
          shad,
          matte,
          mask,
          gmatte,
          valid_points,
          patch_size,
          align_rot,
          align_trans,
          &tempFeaturesMatCv,
          &tempLabelsMatCv);

      // convert the matrices to Eigen
      CvToEigen(tempFeaturesMatCv, tempFeaturesMat);
      CvToEigen(tempLabelsMatCv, tempLabelsMat);
      // count how many rows we've gotten back
      rowCount += static_cast<int>(tempFeaturesMat.rows());
			// resize the matrices to hold enough data
      features->resize(rowCount, tempFeaturesMat.cols());      
			labels->resize(rowCount, tempLabelsMat.cols());
    
      // if the output matrices are empty (the first iteration) simply assign the data to them
			if (!isMatInitialized) {
				*labels << tempLabelsMat;
				isMatInitialized = true;
        *features << tempFeaturesMat;
    	}
      // if the output matrices already contain something, concatenate
      else {
				*labels << prevLabelsMat, tempLabelsMat;
        *features << prevFeaturesMat, tempFeaturesMat;
			}
      // save current state of the output matrices for the next iteration
			prevLabelsMat = *labels;
		  prevFeaturesMat = *features;
		}
	}
}

// assumes that given images are float matrices
void DataProvider::ComputeFeaturesAndLabels(
    const cv::Mat& shad,
    const cv::Mat& matte,
    const cv::Mat& mask,
    const cv::Mat& gmatte,
    const std::vector<Point>& points,
    int patch_size,
    bool align_rot,
    int align_trans,
    cv::Mat* features,
    cv::Mat* labels) {
  Mat features_row;
  Mat labels_row;

  // get the first vector to figure out how many columns there are
  int p(0);
  AlignedPatch aligned_patch(RotatedRect(points[p], Size2f(patch_size, patch_size), 0.f), 0.f);
  features_row = FeaturePatchFromImage(shad, mask, gmatte, align_rot, align_trans, &aligned_patch);
  labels_row = LabelPatchFromImage(matte, aligned_patch);
 
  // initialize the output matrices to the right size
  *features = Mat(points.size(), features_row.cols, IM_TYPE);
  *labels = Mat(points.size(), labels_row.cols, IM_TYPE);
  // copy the first vector into the first matrix row
  features_row.copyTo(features->row(p));
  labels_row.copyTo(labels->row(p));
  
  // get remaining vectors and put them in the output matrices
  for (p = 1; p < points.size(); ++p) {
    // initialize the rot_rect to be at the position points[p]
    aligned_patch.set_rot_rect(RotatedRect(points[p], Size2f(patch_size, patch_size), 0.f));
    aligned_patch.set_intensity_offset(0.f);
    // compute the features (also aligns and expands the rot_rect)
    features_row = FeaturePatchFromImage(shad, mask, gmatte, align_rot, align_trans, &aligned_patch);
    features_row.copyTo(features->row(p));

    // uses the same (aligned and expanded) rot_rect as the feature computation
    labels_row = LabelPatchFromImage(matte, aligned_patch);
    labels_row.copyTo(labels->row(p));

    //Mat intensity_patch(CutOutRotatedRect(shad, rot_rect));
    //Mat matte_patch(CutOutRotatedRect(matte, rot_rect));
    //imshow("intensity", intensity_patch);
    //imshow("matte", matte_patch);
    //waitKey();
    //destroyAllWindows();
	}
}

float rand1() {
  return static_cast<float>(rand())/static_cast<float>(RAND_MAX);
}

void DataProvider::DrawRotatedRect(const RotatedRect& rot_rect, const Scalar& color, Mat* img) {
  Size rect_size(rot_rect.size);
  float angle(rot_rect.angle * 3.14159 / 180.0);
  vector<Point2f> corners;
  corners.push_back(Point(rot_rect.center.x - rot_rect.size.width/2, rot_rect.center.y - rot_rect.size.height/2));
  corners.push_back(Point(rot_rect.center.x + rot_rect.size.width/2, rot_rect.center.y - rot_rect.size.height/2));
  corners.push_back(Point(rot_rect.center.x + rot_rect.size.width/2, rot_rect.center.y + rot_rect.size.height/2));
  corners.push_back(Point(rot_rect.center.x - rot_rect.size.width/2, rot_rect.center.y + rot_rect.size.height/2));
  corners.push_back(Point(rot_rect.center.x, rot_rect.center.y - rot_rect.size.height)); // up point

  Mat rot_mat(getRotationMatrix2D(rot_rect.center, rot_rect.angle, 1.0));
  for (auto cor(corners.begin()); cor != corners.end(); ++cor) {
    *cor -= rot_rect.center;
    float old_x(cor->x);
    float old_y(cor->y);
    cor->x = old_x*cos(angle) - old_y * sin(angle);
    cor->y = old_x*sin(angle) + old_y * cos(angle);
    *cor += rot_rect.center;
  }

  for (int l(0); l < 3; ++l) {
    line(*img, corners[l], corners[l+1], color);
  }
  line(*img, corners[3], corners[0], color);
  line(*img, rot_rect.center, corners[4], color);
}

void DataProvider::Align(
    const cv::Mat& image,
    bool align_rot,
    int align_trans,
    AlignedPatch* aligned_patch) {
  // if both alignment types are off, don't do anything
  if (!align_rot && align_trans == 0) {
    return;
  }

  Mat tmplt(16, 16, IM_TYPE);
  tmplt.setTo(0.f);
  tmplt(Rect(0,0,16,8)).setTo(1.f);
  //imshow("template", tmplt);
  float mean_tmplt(mean(tmplt)[0]);

  double min_dist(DBL_MAX); 
  Point best_point(-1, -1);
  float best_angle(0.f);
  float best_intensity_offset(0.f);

  int center_min(-align_trans);
  int center_max(align_trans);
  int trans_step(1);
  float angle_step(4.f); // degrees
  float max_angle(align_rot ? 360.f : 0.0f);

  //imshow("tmplt", tmplt);
  //imshow("original", CutOutRotatedRect(image, aligned_patch->rot_rect()));

  Mat board;
  Mat temp_board;
  cvtColor(image, board, CV_GRAY2BGR);
  for (int x(center_min); x < center_max + 1; x += trans_step) {
    for (int y(center_min); y < center_max + 1; y += trans_step) {
      for (float angle(0.f); angle < max_angle; angle += angle_step) {
        Point new_center(aligned_patch->rot_rect().center.x + x, aligned_patch->rot_rect().center.y + y);
        //Point new_center(rot_rect->center.x, rot_rect->center.y);
        RotatedRect candidate_rect(new_center, aligned_patch->rot_rect().size, angle);

        //if (angle == 0.f) {
        //  temp_board = board.clone();
        //  DrawRotatedRect(aligned_patch->rot_rect(), Scalar(0.0, 0.0, 255.0), &temp_board);
        //  DrawRotatedRect(RotatedRect(best_point, aligned_patch->rot_rect().size, best_angle), Scalar(0.0, 255.0, 0.0), &temp_board);
        //  DrawRotatedRect(candidate_rect, Scalar(255.0, 0.0, 255.0), &temp_board);
        //  imshow("board", temp_board);
        //  waitKey(1);
        //}

        Mat candidate_patch(CutOutRotatedRect(image, candidate_rect));

        //std::cout << "BEFORE: " << mean(CutOutRotatedRect(image, AlignedPatch(candidate_rect, 0.f)))[0] << std::endl;
        //imshow("before", image);
        
        // normalize
        float mean_cand(mean(candidate_patch)[0]);
        float intensity_offset(mean_tmplt - mean_cand);
        //// DBG
        //intensity_offset = 0.f;
        candidate_patch += intensity_offset;
        
        //std::cout << "AFTER: " << mean(CutOutRotatedRect(image, AlignedPatch(candidate_rect, 0.f)))[0] << std::endl;
        //imshow("after", image);
        //waitKey();
        //destroyAllWindows();

        double dist(cv::norm(tmplt, candidate_patch, NORM_L2));
        if (dist < min_dist) {
          if (abs(dist - min_dist) < 0.01) {
            // if the dists are almost the same, take one with smallest offset
            float new_offset(Vector2f(aligned_patch->rot_rect().center.x - new_center.x, aligned_patch->rot_rect().center.y - new_center.y).norm());
            float old_offset(Vector2f(aligned_patch->rot_rect().center.x - best_point.x, aligned_patch->rot_rect().center.y - best_point.y).norm());
            if (new_offset <= old_offset) {
              min_dist = dist;
              best_angle = angle;
              best_point = new_center;
              best_intensity_offset = intensity_offset;
            }
          } else {
            // otherwise simply take the new one
            min_dist = dist;
            best_angle = angle;
            best_point = new_center;
            best_intensity_offset = intensity_offset;
          }
          //imshow("candidate", candidate_patch);
        }
      }
    }
  }
  
  // DBG
  imshow("tmplt", tmplt);
  imshow("original", CutOutRotatedRect(image, aligned_patch->rot_rect()));

  aligned_patch->rot_rect_ptr()->center = best_point;
  aligned_patch->rot_rect_ptr()->angle = best_angle;
  aligned_patch->set_intensity_offset(best_intensity_offset);

  // DBG
  imshow("aligned", CutOutRotatedRect(image, aligned_patch->rot_rect()) + aligned_patch->intensity_offset());
  waitKey();
  destroyAllWindows();

  return;
}

Mat DataProvider::Unalign(
    const cv::Mat& image_patch,
    const RotatedRectOffset& original_offset,
    RotatedRect* unaligned) {
  // center of the big window
  Point center(image_patch.cols/2, image_patch. rows/2);
  //// center of the sub-window
  //float beta(atan(original_offset.offset().x / original_offset.offset().y));
  //float gamma(original_offset.angle() * PI/180.f + beta);
  //float d(sqrt(original_offset.offset().x * original_offset.offset().x + original_offset.offset().y * original_offset.offset().y));
  ////gamma = gamma * PI/180.f;
  //Point2f sub_center(center.x + d * sin(gamma), center.y + d * cos(gamma));

  //// create cutout_Rect
  //RotatedRect cutout_rect(
  //    sub_center,
  //    Size2f(16, 16),
  //    -original_offset.angle());

  //if (unaligned != NULL) {
  //  //unaligned->center.x += original_offset.offset().x;
  //  //unaligned->center.y += original_offset.offset().y;
  //  //unaligned->angle -= original_offset.angle();
  //  //unaligned->size = cutout_rect.size;

  //  Mat board(image_patch);
  //  cvtColor(board, board, CV_GRAY2BGR);
  //  DrawRotatedRect(cutout_rect, Scalar(0.0, 0.0, 255.0), &board);
  //  imshow("small board", board);
  //}

  //Mat cutout(CutOutRotatedRect(image_patch, cutout_rect));
  //cutout -= original_offset.intensity_offset();
  //return cutout;

  //imshow("window", image_patch);
  // create a rotated copy of the big window
  Mat rot_mat(getRotationMatrix2D(center, original_offset.angle(), 1.0));
  Mat rotated;
  warpAffine(image_patch.clone(), rotated, rot_mat, image_patch.size(), WARP_INVERSE_MAP | INTER_CUBIC);
  //// take the mean intensty back to the original state
  //rotated -= original_offset.intensity_offset();
  //imshow("rotated", rotated);
  
  // get initial subpatch
  RotatedRect cutout_rect(Point(image_patch.cols/2, image_patch.rows/2), image_patch.size(), 0.f);
  // shrink it
  ShrinkRotRect(PATCH_FEATURE_EXTENSION, &cutout_rect);
  // translate it
  cutout_rect.center.x -= original_offset.offset().x;
  cutout_rect.center.y -= original_offset.offset().y;
  // return the resulting cutout
  Mat cutout(CutOutRotatedRect(rotated, cutout_rect));
  cutout -= original_offset.intensity_offset();

  //DrawRotatedRect(cutout_rect, Scalar(0.0, 0.0, 255.0), &rotated);
  //Mat rotated_copy(rotated.clone());
  //warpAffine(rotated.clone(), rotated_copy, rot_mat, image_patch.size(), INTER_CUBIC);
  //imshow("final", rotated_copy);

  return cutout;
}

cv::Point2f DataProvider::RotatePoint(const cv::Point2f& point, const float& angle, const cv::Point2f& pivot) {
  Point2f rot_point;
  // offset the point by the pivot
  float offset_x(point.x - pivot.x);
  float offset_y(point.y - pivot.y);
  // rotate by angle
  float cos_alpha(cos(angle));
  float sin_alpha(sin(angle));
  rot_point.x = offset_x * cos_alpha - offset_y * sin_alpha;
  rot_point.y = offset_x * sin_alpha + offset_y * cos_alpha;
  // offset back
  rot_point.x += pivot.x;
  rot_point.y += pivot.y;
  return rot_point;
}


cv::Mat DataProvider::CutOutRotatedRect(const cv::Mat& image, const RotatedRect& rot_rect) {
  // get the image region inside the bounding box of the rot_rect
  Rect bounding(rot_rect.boundingRect());
  // clone to make sure we create deep copy
  cv::Mat image_rot(image(bounding).clone());
  Mat cutout;

  if (rot_rect.angle == 0.0f) {
    cutout = image_rot;
  } else {
    // rotate the region back by rot_rect.angle
    const cv::Point2f center(static_cast<float>(image_rot.cols/2), static_cast<float>(image_rot.rows/2));
    Mat rot_mat(getRotationMatrix2D(center, -rot_rect.angle, 1.0));
    warpAffine(image_rot, cutout, rot_mat, image_rot.size(), WARP_INVERSE_MAP | INTER_CUBIC); 
    // cut out the final region of the size rot_rect.size
  }
  // add intensity offset
  //std::cout << "BEFORE ADD " << mean(cutout)[0] << std::endl;
  //cutout += aligned_patch.intensity_offset();
  //std::cout << "AFTER ADD " << mean(cutout)[0] << std::endl;

  cv::Point top_left((cutout.cols - rot_rect.size.width)/2, (cutout.rows - rot_rect.size.height)/2);
  return cutout(cv::Rect(top_left, rot_rect.size)); 
}

cv::Mat DataProvider::FeaturePatchFromImage(
    const cv::Mat& image,
    const cv::Mat& mask,
    const cv::Mat& gmatte,
    bool align_rot,
    int align_trans,
    AlignedPatch* aligned_patch) {
  //Mat check_gray(imreadFloat("C:\\Users\\mgryka\\Desktop\\brick_wall.png", 0));
  //Mat check;
  //cvtColor(check_gray, check, CV_GRAY2BGR);
  //int b(50);
  //int w(250);

  //for (int ir(0); ir < 100; ++ir) {
  //  Point rpos(b + w*rand1(), b + w*rand1());
  //  RotatedRect crr(rpos, Size(16, 16), 0.f);

  //  //DrawRotatedRect(crr, &check);
  //  //imshow("check", check); 
  //  //waitKey();
  //  //destroyAllWindows();

  //  DrawRotatedRect(crr, Scalar(0.0, 0.0, 255.0), &check);
  //  Align(check_gray, &crr);
  //  DrawRotatedRect(crr, Scalar(0.0, 255.0, 0.0), &check);
  //  imshow("board", check);
  //  waitKey();
  //}
  //destroyAllWindows();

  Align(image, align_rot, align_trans, aligned_patch);
  // expand the rect to compute (and store) features of a larger window
  ExpandRotRect(PATCH_FEATURE_EXTENSION, aligned_patch->rot_rect_ptr());
  return GetFeatureVector(image, mask, gmatte, *aligned_patch);
}

cv::Mat DataProvider::LabelPatchFromImage(
    const cv::Mat& image,
    const AlignedPatch& aligned_patch) {
  return GetLabelVector(image, aligned_patch);
}

Mat DataProvider::GetFeatureVector(
    const cv::Mat& image,
    const cv::Mat& mask,
    const cv::Mat& gmatte, 
    const AlignedPatch& aligned_patch) {
  // cut out the (rotated) patch from the image
  Mat patch(CutOutRotatedRect(image, aligned_patch.rot_rect()));
  patch += aligned_patch.intensity_offset();
  
  if (!IsInImage(aligned_patch.rot_rect().boundingRect(), image)) {
    std::cerr << "image too small for the expanded patch!" << std::endl;
    std::cerr << "image size: " << image.cols << "x" << image.rows 
        << "; patch: (" << aligned_patch.rot_rect().boundingRect().x << ", " << aligned_patch.rot_rect().boundingRect().y << ") - (" 
        << aligned_patch.rot_rect().boundingRect().x + aligned_patch.rot_rect().boundingRect().width << ", " << aligned_patch.rot_rect().boundingRect().y + aligned_patch.rot_rect().boundingRect().height
        << ")" << std::endl;
    throw;
  }
	assert(patch.channels() == 1);

  //imshow("intensity patc5h", patch);

  map<string, Mat> featureMap;
  // IMPORTANT: remember to flatten the features (reshape(0, 1))
  if (active_features_["feature_intensity"]) {
    featureMap["feature_intensity"] = GetFeatureIntensity(patch);
  }
  if (active_features_["feature_gradient_orientation"]) {
    featureMap["feature_gradient_orientation"] = GetFeatureGradientOrientation(patch);
  }
  if (active_features_["feature_gradient_magnitude"]) {
    featureMap["feature_gradient_magnitude"] = GetFeatureGradientMagnitude(patch);
  }
  if (active_features_["feature_gradient_xy"]) {
    featureMap["feature_gradient_xy"] = GetFeatureGradientXY(patch);
  }
  if (active_features_["feature_distance_transform"]) {
    featureMap["feature_distance_transform"] = GetFeatureDistanceTransform(mask, aligned_patch.rot_rect().center);
  }
  if (active_features_["feature_polar_angle"]) {
    featureMap["feature_polar_angle"] = GetFeaturePolarAngle(mask, aligned_patch.rot_rect().center);
  }
  if (active_features_["feature_gmatte"]) {
    featureMap["feature_gmatte"] = GetFeatureGmatte(gmatte, aligned_patch);
  }

	int n_features(0);
  for (auto it(featureMap.begin()); it != featureMap.end(); ++it) {
    n_features += it->second.cols;
  }

	Mat features(1, n_features, IM_TYPE);
	int pos(0);
  for (auto it(featureMap.begin()); it != featureMap.end(); ++it) {
    pos = CopyIntoVector(it->second, features, pos);
  }

	return features;
}

Mat DataProvider::GetLabelVector(const cv::Mat& image, const AlignedPatch& aligned_patch) {
  Mat label_vector(CutOutRotatedRect(image, aligned_patch.rot_rect()).clone().reshape(0, 1));
  label_vector += aligned_patch.intensity_offset();
  return label_vector;
}

Mat DataProvider::imreadFloat(const string& path, int channel) {
	if (!FileExists(path)) {
		cerr << "File \'" << path << "\' cannot be read." << endl;
		throw;
	}
	Mat im(imread(path));
	im.convertTo(im, CV_32F, 1.0/255.0);

	if (channel == ALL) {
		return im;
  }
  //else if (channel == LIGHTNESS) {
	 // vector<Mat> imChannels(3);
  //  cvtColor(im, im, CV_RGB2Lab);
	 // split(im, imChannels);
  //  return 
  //} 
  else {
	  vector<Mat> imChannels(3);
	  split(im, imChannels);
	  return imChannels[channel];
  }
}

// get gradient of im in X (x = true) or Y (x = false) direction
Mat DataProvider::GetGradient(const Mat& im, bool x, bool sobel) {
	Mat kernel(Mat::zeros(3, 3, IM_TYPE));

	if (sobel) {
		if (x) {
			kernel.at<float>(0, 0) = -1.0;
			kernel.at<float>(0, 2) =  1.0;
			kernel.at<float>(1, 0) = -2.0;
			kernel.at<float>(1, 2) =  2.0;
			kernel.at<float>(2, 0) = -1.0;
			kernel.at<float>(2, 2) =  1.0;
		} else {
			kernel.at<float>(0, 0) = -1.0;
			kernel.at<float>(0, 1) = -2.0;
			kernel.at<float>(0, 2) = -1.0;
			kernel.at<float>(2, 0) =  1.0;
			kernel.at<float>(2, 1) =  2.0;
			kernel.at<float>(2, 2) =  1.0;
		}
	} else {
		if (x) {
			kernel.at<float>(1, 0) = -1.0;
			kernel.at<float>(1, 2) = 1.0;
		} else {
			kernel.at<float>(0, 1) = -1.0;
			kernel.at<float>(2, 1) = 1.0;
		}
	}

	Mat grad;
	filter2D(im, grad, IM_TYPE, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

	return grad;
}

Mat DataProvider::GetFeatureIntensity(const cv::Mat& im) {
  Mat downscaled(im);
  Downsample(&downscaled, 0);
  return downscaled.reshape(0, 1);
}

Mat DataProvider::GetFeatureGradientXY(const cv::Mat& im) {
  Mat downscaled(im);
  Downsample(&downscaled);
  Mat grad_x(GetGradient(downscaled, true).reshape(0,1));
	Mat grad_y(GetGradient(downscaled, false).reshape(0,1));

  Mat grad_xy(1, grad_x.cols + grad_y.cols, IM_TYPE);
  int pos(0);
  pos = CopyIntoVector(grad_x, grad_xy, pos);
  pos = CopyIntoVector(grad_y, grad_xy, pos);
  return grad_xy;
}

Mat DataProvider::GetFeatureGradientOrientation(const cv::Mat& im) {
  Mat downscaled(im);
  Downsample(&downscaled);
	Mat grad_x(GetGradient(downscaled, true));
	Mat grad_y(GetGradient(downscaled, false));
	
	Mat out(downscaled.rows, downscaled.cols, IM_TYPE);
	// TODO: is there a faster way to compute the atan2 for the whole matrix (grad_y/grad_x)?
	for (int y = 0; y < out.rows; y++) {
		for (int x = 0; x < out.cols; x++) {
			out.at<float>(y, x) = cv::fastAtan2(grad_y.at<float>(y, x), grad_x.at<float>(y, x));
		}
	}

	// convert to radians
	out *= PI_180;

	return out.reshape(0, 1);;
};

Mat DataProvider::GetFeatureGradientMagnitude(const cv::Mat& im) {
  Mat downscaled(im);
	Mat grad_x(GetGradient(downscaled, true));
	Mat grad_y(GetGradient(downscaled, false));
	
  // TODO: is it ok not to sqrt?
  return grad_y.mul(grad_y) + grad_x.mul(grad_x);
}

// TODO: broken
cv::Mat DataProvider::GetFeaturePlaneNormal(const cv::Mat& patch) {
  //EigenMat patchEigen;
  //DataProvider::CvToEigen(patch, patchEigen);
  //NormalType normal = PlaneFitter::fitPlaneRansacGetNormal(patchEigen);

  Mat normalCv;
  //eigen2cv(normal, normalCv);
  return normalCv.reshape(0, 1);;
}

// TODO: this is very wasteful - we're computing distance transform for the same image many times over
// should do caching instead. How to detect image changegi?
cv::Mat DataProvider::GetFeatureDistanceTransform(const cv::Mat& mask, const cv::Point& pos) {
  Mat dtrans;
  Mat mask_conv;
  mask.convertTo(mask_conv, CV_8UC1, 255.0);
  distanceTransform(mask_conv, dtrans, CV_DIST_L2, 3);
  normalize(dtrans, dtrans, 0.0, 1.0, NORM_MINMAX);
  Mat feature(1,1, CV_32F);
  feature.setTo(dtrans.at<float>(pos));
  return feature;
}

// TODO: efficiency - see getFeatureDistanceTransform above
cv::Mat DataProvider::GetFeaturePolarAngle(const cv::Mat& mask, const cv::Point& pos) {
  Mat dtrans;
  Mat mask_conv;
  mask.convertTo(mask_conv, CV_8UC1, 255.0);
  distanceTransform(mask_conv, dtrans, CV_DIST_L2, 3);
  Point max_loc;
  minMaxLoc(dtrans, NULL, NULL, NULL, &max_loc);
  Vec2f diff(static_cast<Point2f>(pos - max_loc));
  cv::Mat angle(1, 1, CV_32F);
  angle.setTo(fastAtan2(diff(1), diff(0)));
  return angle;
}

cv::Mat DataProvider::GetFeatureGmatte(const cv::Mat& gmatte, const AlignedPatch& aligned_patch) {
  Mat cutout(CutOutRotatedRect(gmatte, aligned_patch.rot_rect()));
  //.clone().reshape(0, 1));
  Downsample(&cutout, 0);
  return cutout.reshape(0, 1);
}

cv::Mat DataProvider::DownsamplePatch(const cv::Mat& patch, int times) {
  Mat downPatch(patch);
  
  for (; times > 0; --times) {
    pyrDown(downPatch, downPatch);
  }
  
  return downPatch;
}

void DataProvider::serializeEigenMatAscii(const EigenMat& data, const string& fileName, bool write_dimensions) {
  int rows(static_cast<int>(data.rows()));
  int cols(static_cast<int>(data.cols()));

  // open the file for writing
  ofstream opFile;
  opFile.open(fileName, ios::out);
  
  if (write_dimensions) {
    // write two integers for number of rows and columns
    opFile << rows << ",";
    opFile << cols << "\n";
  }

  float temp;
	for (int rIt = 0; rIt < rows; rIt ++) {
		for (int cIt = 0; cIt < cols; cIt ++) {
      temp = data(rIt, cIt);
      opFile << temp;
      if (cIt < cols-1) opFile << ",";
		}
    opFile << "\n";
	}

	opFile.close();
}

EigenMat DataProvider::deserializeEigenMatAscii(const string& file_name) {
  std::vector<std::vector<float> > floats;
  std::ifstream  data(file_name);
  std::string line;
  int rows(0);
  int cols(0);
  while(std::getline(data,line)) {
    floats.push_back(std::vector<float>());
    std::stringstream lineStream(line);
    std::string cell;
    while(std::getline(lineStream,cell,',')) {
      floats[rows].push_back(atof(cell.c_str()));
    }
    ++rows;
  }
  cols = floats[0].size();
  EigenMat mat(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      mat(r, c) = floats[r][c];
    }
  }
  return mat;
}

void DataProvider::Downsample(cv::Mat* patch, int interpolation) {
  assert(patch->rows == patch->cols);
  cv::resize(*patch, *patch, Size(MAX_PATCH_SIZE, MAX_PATCH_SIZE), 0.0, 0.0, interpolation);
  //while (patch->rows > MAX_PATCH_SIZE) {
    //pyrDown(*patch, *patch);
  //}
}
