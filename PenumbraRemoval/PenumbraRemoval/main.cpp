#include <ctime>

#include <iostream>
#include <string>
#include <regex>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <DataProvider/DataProvider.h>
#include <DataProvider/PCAWrapper.h>
//#include <Clusterer.h>
#include <Ensemble.h>
#include <TreeR.h>
#include <NodeR.h>
#include <EnsembleParams.h>


#include <Regularization/RegLabel.h>

#include "PenumbraRemover.h"
#include "OptionParser.h"
#include "Options.h"

#include <DataSearcher.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define EIGEN_DEFAULT_TO_ROW_MAJOR

using namespace std;

DataProvider::CHANNEL g_active_channel = DataProvider::RED;

std::string vectorString(Eigen::RowVectorXf vec) {
	std::stringstream ss;

	std::string s;
	int cols = int(vec.cols());
	for (int c = 0; c < cols; c++) {
		ss << vec(c);
		if (c < cols - 1) {
			ss << ",";
		}

	}
	ss >> s;
	return s;
}

EnsembleParams getEnsembleParams(const Options& options) {
	EnsembleParams ep(
      options.GetParamInt("nDimIn"),
		  options.GetParamInt("nDimOut"),
		  options.GetParamInt("nTrees"),
		  options.GetParamInt("treeDepth"),
		  options.GetParamInt("nDimTrials"),
		  options.GetParamInt("nThreshTrials"),
		  options.GetParamFloat("bagProb"),
		  options.GetParamInt("minExsAtNode"));
	return ep;
}

void train(const Options& options) {
	int finestScale(options.GetParamInt("finest_scale"));
	int nScales(options.GetParamInt("n_scales"));
  int scaleStep(options.GetParamInt("scale_step"));
	vector<string> imageNames(DataProvider::getLinesAsStrVec(options.GetParamString("image_list_file")));

  PenumbraRemover prem(finestScale, nScales, scaleStep, 0.0);
  int s = finestScale;
  // if finest scale is supposed to have uniform labels, don't train it
  if (options.GetParamInt("uniform_finest")) {
    s += scaleStep;
  }
	for (; s < finestScale + nScales * scaleStep; s += scaleStep) {
		cout << "loading data at scale " << s << "... ";
    // create scale-dependent file names
    char dataFile[PATH_LENGTH], labelsFile[PATH_LENGTH];
		sprintf(dataFile, options.GetParamString("data_file").c_str(), s);
		sprintf(labelsFile, options.GetParamString("labels_file").c_str(), s);

    EigenMat data;
    EigenMat labels;
    if (options.GetParamInt("generate_data")) {
      std::cout << "generating... ";
		  vector<ImageTriple> imageTriples(DataProvider::getImagePaths(options.GetParamString("image_folder"), imageNames));

      //QualifyImages

      DataProvider dp;
      map<string, bool> activeFeatures = options.GetActiveFeatures();
      dp.set_active_features(activeFeatures);

      //EigenMat dummy(5, 5);
      //dummy.setRandom();
      //std::cout << dummy << std::endl;
      //dummy = dummy.block(0, 0, 4, 4);
      //std::cout << dummy << std::endl;
      //dummy = dummy.block(0, 0, 2, 2);
      //dummy.res
      //std::cout << dummy << std::endl;


    //  // DBG with training data clustering
    //  EigenMat centers = dp.deserializeEigenMat("C:\\Work\\research\\shadow_removal\\experiments\\output\\size200\\data\\centers.dat");
    //  std::vector<std::vector<int> > cluster_membership = dp.GetClusteredData(imageTriples, s, g_active_channel, 25, &centers, &labels, &data);
    //  //data = DataProvider::deserializeEigenMat(string(dataFile));
		  ////labels = DataProvider::deserializeEigenMat(string(labelsFile));

      //// DBG
      //std::vector<ImageTriple> imtv_short;
      //imtv_short.push_back(imageTriples[0]);
      //dp.GetLabelSubset(imtv_short, s, g_active_channel, 10, &labels, &data);

      // without training data clustering
      int n_labels = options.GetParamInt("n_training_samples");
      clock_t t0(clock());
      dp.GetLabelSubset(imageTriples, s, g_active_channel, n_labels, options.GetParamBool("align_rot"), options.GetParamInt("align_trans"), &labels, &data);
      std::cout << "FEATURE EXTRACTION TIME: " << static_cast<double>(clock() - t0)/CLOCKS_PER_SEC << std::endl;

      //// DBG: write out intensity and label patches
      //std::string out_path("C:\\Work\\research\\shadow_removal\\experiments\\output\\size2\\patches\\");
      //for (int r(0); r < data.rows(); ++r) {
      //  EigenMat intensity(data.row(r));
      //  EigenMat matte(labels.row(r));
      //  cv::Mat intensity_cv;
      //  cv::Mat matte_cv;
      //  eigen2cv(intensity, intensity_cv);
      //  eigen2cv(matte, matte_cv);
      //  intensity_cv = intensity_cv.reshape(0, 32);
      //  matte_cv = matte_cv.reshape(0, 32);
      //  std::stringstream ss_f;
      //  ss_f << out_path << "\\features\\" << r << ".png";
      //  imwrite(ss_f.str(), intensity_cv*255);
      //  std::stringstream ss_l;
      //  ss_l << out_path << "\\labels\\" << r << ".png";
      //  imwrite(ss_l.str(), matte_cv*255);
      //}


      DataProvider::serializeEigenMat<float>(data, string(dataFile));
		  DataProvider::serializeEigenMat<float>(labels, string(labelsFile), false);
      std::cout << "got label subset" << std::endl;

    //  // DBG
    //  labels = DataProvider::deserializeEigenMat(string(labelsFile));
    //  int patch_size = DataProvider::getPatchSize(s);
    //  for (int r = 0; r < 10; ++r) {
    //    EigenMat patch = labels.row(r);
    //    cv::Mat patch_cv;
    //    cv::eigen2cv(patch, patch_cv);
    //    cv::imwrite("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\patches\\patch_long.png", patch_cv*255);
			 // patch.resize(patch_size, patch_size);
    //    cv::eigen2cv(patch, patch_cv);
    //    cv::imwrite("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\patches\\patch.png", patch_cv*255);
			 // patch.transposeInPlace();
    //    cv::eigen2cv(patch, patch_cv);
    //    cv::imwrite("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\patches\\patch_t.png", patch_cv*255);
    //    //std::cout << patches[r] << std::endl;
    //    //std::cout << patch_cv << std::endl;
		  //}

    } else {
      std::cout << "reading... ";
      data = DataProvider::deserializeEigenMat<float>(dataFile);
      labels = DataProvider::deserializeEigenMat<float>(labelsFile, false);
    }
		cout << "done" << endl;

    string ensembleFile(DataProvider::insertScaleNum(options.GetParamString("ensemble_file"), s));
		string pcaFile(DataProvider::insertScaleNum(options.GetParamString("pca_file"), s));

		cout << "training scale " << s << " on " << data.rows() << " patches ... " << endl;

    bool compute_pca = options.GetParamInt("compute_pca") == 1;
    std::string leaves_path = "";//"G:\\projects\\penumbraRemoval\\2012-06-29\\output\\size1861\\leaves\\";

    prem.TrainScale(s, data, labels, options, ensembleFile, pcaFile, compute_pca, leaves_path);

		//cout << "testing projection... " << endl;
  //  prem.TestProjection(s, labels, pcaFile, "C:\\Work\\VS2010\\PenumbraRemoval\\x64\\data\\tree_patches\\");

		cout << "\rdone\t\t\t\t\t\t\t\t" << endl;
	}
}

std::string IntToString(const int& val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

void writeout_node_splitdim(RandomForest::NodeR* node, std::ofstream& outfile) {
  if (!node->is_leaf()) {
    outfile << node->split_dim() << std::endl;
    writeout_node_splitdim(node->left(), outfile);
    writeout_node_splitdim(node->right(), outfile);
  }
}

void main(int argc, char* argv[]) {

  //RandomForest::Ensemble ensemble;
  //ensemble.ReadEnsembleBin("C:\\Work\\research\\writing\\SoftShadows\\paper2013b\\data\\trees4.dat");
  ////ensemble.WriteEnsemble("C:\\Work\\research\\writing\\SoftShadows\\paper2013b\\data\\trees4.txt");

  //std::ofstream outfile("C:\\Work\\research\\writing\\SoftShadows\\paper2013b\\data\\split_dims.txt");

  //for (int t(0); t < ensemble.n_trees(); ++t) {
  //  //RandomForest::TreeR* tree = ensemble.trees()[t];
  //  RandomForest::NodeR* current_node = ensemble.trees()[t]->root();
  //  writeout_node_splitdim(current_node, outfile);
  //}

  //outfile.close();

  // TODO: randomize
  srand(12345);
  //srand(unsigned(time(NULL)));

	cout << "reading options file " << argv[1] << "... ";
	OptionParser op(argv[1]);
  Options options(op.options());
	cout << "done" << endl;

  if (options.GetParamInt("train")) {
		train(options);
	}

	if (options.GetParamInt("test")) {
        //{
        //  char labels_file[PATH_LENGTH];
        //  char features_file[PATH_LENGTH];
        //  sprintf(labels_file, options.GetParamString("labels_file").c_str(), 4);
        //  sprintf(features_file, options.GetParamString("data_file").c_str(), 4);
        //  EigenMat labels(DataProvider::deserializeEigenMat(labels_file));
        //  EigenMat features(DataProvider::deserializeEigenMat(features_file));
        //  EigenMat label;
        //  EigenMat feature;
        //  cv::Mat label_cv;
        //  cv::Mat feature_cv;
        //  int patchSize(32);
        //  std::string label_path("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\patches\\labels\\patch%d.png");
        //  std::string feature_path("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\patches\\features\\patch%d.png");
        //  PCAWrapper pcaw("C:\\Work\\research\\shadow_removal\\experiments\\output\\size1861\\data\\pcaw4.xml");
        //  for (int r = 0; r < labels.rows(); ++r) {
        //    label = labels.row(r);
        //    feature = features.row(r).block(0, 0, 1, 256);
        //    label.resize(patchSize, patchSize);
        //    feature.resize(16, 16);
        //    label.transposeInPlace();
        //    feature.transposeInPlace();
        //    eigen2cv(label, label_cv);
        //    eigen2cv(feature, feature_cv);
        //    label_cv *= 255.0;
        //    feature_cv *= 255.0;
        //    imwrite(DataProvider::insertScaleNum(label_path, r), label_cv);
        //    imwrite(DataProvider::insertScaleNum(feature_path, r), feature_cv);
        //  }
        //}
  	    cout << "reading image list " << options.GetParamString("image_list_file") << "... ";
	      vector<string> imageNames(DataProvider::getLinesAsStrVec(options.GetParamString("image_list_file")));
	      cout << "done" << endl;

        PenumbraRemover pr(options.GetParamInt("finest_scale"), options.GetParamInt("n_scales"), options.GetParamInt("scale_step"), options.GetParamFloat("unary_cost_scaling"));
        RegLabel::setWeights(options.GetParamFloat("relationship_weight_peer"), options.GetParamFloat("relationship_weight_parent"), options.GetParamFloat("pairwise_weight_beta"));

        // give pr the training labels so that it can get actual samples during regularization
        char labelsFile[PATH_LENGTH];
        std::map<int, EigenMat> labels;

        int finest_scale = options.GetParamInt("finest_scale");
        int n_scales = options.GetParamInt("n_scales");
        int scale_step = options.GetParamInt("scale_step");

        int s = finest_scale;
        // if finest scale is supposed to have uniform labels, don't read it
        if (options.GetParamInt("uniform_finest")) {
          s += scale_step;
        }
        for (; s < finest_scale + n_scales * scale_step; s += scale_step) {
          sprintf(labelsFile, options.GetParamString("labels_file").c_str(), s);
          labels[s] = DataProvider::deserializeEigenMat<float>(labelsFile, false);

          pr.set_labels(s, &(labels[s]));
          //// DBG
          //pr.TestProjection(s, labels[s], DataProvider::insertScaleNum(options.GetParamString("pcaFile"), s), "C:\\Work\\research\\shadow_removal\\experiments\\output\\size100\\labels\\");
        }

        map<string, bool> activeFeatures = options.GetActiveFeatures();
        pr.set_active_features(activeFeatures);

        pr.set_data_file(options.GetParamString("data_file"));

        pr.set_unary_cost(options.GetParamInt("unary_cost"));
        pr.set_uniform_finest(options.GetParamInt("uniform_finest"));
        for (int f = 0; f < imageNames.size(); ++f) {
            options.SetParam("image_name", imageNames[f]);
            cout << "testing " << imageNames[f] << endl;
            // read in shadow and mask images
            string shadImPath = DataProvider::getImageFileName(
            options.GetParamString("image_folder") + "\\" + imageNames[f],
            DataProvider::SHAD);

            string maskImPath = DataProvider::getImageFileName(
            options.GetParamString("image_folder") + "\\" + imageNames[f],
            DataProvider::MASK);

            string unshadMaskImPath = DataProvider::getImageFileName(
                options.GetParamString("image_folder") + "\\" + imageNames[f],
                DataProvider::UNSHAD_MASK);

            string gmatteImPath = DataProvider::getImageFileName(
                options.GetParamString("image_folder") + "\\" + imageNames[f],
                DataProvider::GMATTE);

            string matteImPath = DataProvider::getImageFileName(
                options.GetParamString("image_folder") + "\\" + imageNames[f],
                DataProvider::MATTE_GT);

            cv::Mat testShadIm(DataProvider::imreadFloat(shadImPath, g_active_channel));
            cv::Mat testShadImColor(DataProvider::imreadFloat(shadImPath, DataProvider::ALL));

            cv::Mat testMaskIm(DataProvider::imreadFloat(maskImPath, g_active_channel));
            cv::Mat testUnshadMaskIm(DataProvider::imreadFloat(unshadMaskImPath, g_active_channel));
            cv::Mat testGmatteIm;
            if (options.GetParamInt("plane_inpaint") == 0) {
                testGmatteIm = cv::Mat(DataProvider::imreadFloat(gmatteImPath, g_active_channel));
            }

            // DBG
            if (options.IsParamValid("grid_offset") && options.GetParamBool("grid_offset")) {
                assert(options.IsParamValid("grid_offset_x") && options.IsParamValid("grid_offset_y"));
                const int patch_size(16);
                // make the image one patch_size smaller in each dimension and start cutting it at (new_x, new_y) offset
                int new_width(testShadIm.cols - patch_size);
                int new_height(testShadIm.rows - patch_size);

                int new_x(options.GetParamInt("grid_offset_x"));
                int new_y(options.GetParamInt("grid_offset_x"));

                testShadIm = testShadIm(cv::Rect(new_x, new_y, new_width, new_height));
                testMaskIm = testMaskIm(cv::Rect(new_x, new_y, new_width, new_height));
                testUnshadMaskIm = testUnshadMaskIm(cv::Rect(new_x, new_y, new_width, new_height));
            }

            cv::Mat testMatteIm;
            // // dilate the mask to compensate for user not masking enough of the shadow
            // dilate(testMaskIm, testMaskIm, cv::Mat(), cv::Point(-1, -1), 25);
            // expand images in case we need to compute features from area larger than label patches
            int margin = PATCH_MARGIN;
            cv::copyMakeBorder(testShadIm, testShadIm, margin, margin, margin, margin, cv::BORDER_REFLECT);
            cv::copyMakeBorder(testMaskIm, testMaskIm, margin, margin, margin, margin, cv::BORDER_CONSTANT, 0.0f);
            if (options.GetParamInt("plane_inpaint") == 0) {
                cv::copyMakeBorder(testGmatteIm, testGmatteIm, margin, margin, margin, margin, cv::BORDER_REFLECT, 0.0f);
            }
            cv::copyMakeBorder(testUnshadMaskIm, testUnshadMaskIm, margin, margin, margin, margin, cv::BORDER_CONSTANT, 0.0f);
            // calculate cropping params
            cv::Size original_size(testShadIm.cols - 2*margin, testShadIm.rows - 2*margin);
            cv::Rect crop_region(margin, margin, original_size.width, original_size.height);

	        // retrieve matte image
            options.SetParam("original_width", IntToString(original_size.width));
            options.SetParam("original_height", IntToString(original_size.height));
            options.SetParam("margin", IntToString(margin));
            options.SetParam("current_image_name", imageNames[f]);
            cv::Mat matteIm(pr.Test(testShadIm, testMaskIm, testGmatteIm, testUnshadMaskIm, options));

      //cv::Mat matteIm(testShadIm);
      // if we're on Windows, run GradientShop deblocking to get rid of grid arefacts
//#ifdef _WIN32
//      matteImPath = DataProvider::getImageFileName(
//          options.GetParamString("results_dir") + "\\" + imageNames[f],
//          DataProvider::MATTE);
//
//      imwrite(matteImPath + ".jpg", matteIm*255.0);
//      STARTUPINFO info={sizeof(info)};
//      PROCESS_INFORMATION processInfo;
//      std::string command("GS\\GS_Basic.exe " + options.GetParamString("data_dir") + "\\" + "deblock-opt.txt imgFN=" + matteImPath + ".jpg");
//      if (CreateProcess(NULL, LPSTR(command.c_str()), NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
//      {
//          ::WaitForSingleObject(processInfo.hProcess, INFINITE);
//          CloseHandle(processInfo.hProcess);
//          CloseHandle(processInfo.hThread);
//      }
//      std::regex e(imageNames[f]);
//      matteImPath = std::regex_replace(matteImPath, e, "result-" + imageNames[f]);
//      matteIm = DataProvider::imreadFloat(matteImPath + ".tif", g_active_channel);
//      cv::flip(matteIm, matteIm, 0);
//#endif

            //cv::Mat matteIm(testShadIm.size(), testShadIm.type());
            //matteIm.setTo(1.0f);
            //matteIm += pr.Test(testShadIm, testMaskIm, options.GetParamString("ensembleFile"), options.GetParamString("pcaFile"), testMatteIm, cov_im) * testMaskIm;
            //cv::imshow("matte", matteIm);
            //cv::waitKey();
            //cv::destroyAllWindows();

            // crop to original image size
            matteIm(crop_region).copyTo(matteIm);
		    // testShadIm(crop_region).copyTo(testShadIm);
		    // // reconstruct unshadowed image
		    // cv::Mat unshadIm;
            //   //// repeat the retrieved matte for each channel to create a 'color' one
            //   //std::vector<cv::Mat> matte_ch;
            //   //matte_ch.push_back(matteIm);
            //   //matte_ch.push_back(matteIm);
            //   //matte_ch.push_back(matteIm);
            //   //cv::Mat matte_color;
            //   //cv::merge(matte_ch, matte_color);

            //   // DBG postprocess the matte in image space
            //   //cv::GaussianBlur(matteIm, matteIm, cv::Size(51, 51), 0, 0, cv::BORDER_REPLICATE);

            //cv::divide(testShadIm, matteIm, unshadIm);
            ////cv::divide(testShadImColor, matte_color, unshadIm);

            //// write matte and unshadowed to disk
            //string unshadImPath = DataProvider::getImageFileName(
            //       options.GetParamString("results_dir") + "\\" + imageNames[f],
            //       DataProvider::UNSHAD);
            matteImPath = DataProvider::getImageFileName(
                options.GetParamString("results_dir") + "\\" + imageNames[f],
                DataProvider::MATTE);

	        //unshadIm.convertTo(unshadIm, CV_32F, 255.0);
		    matteIm.convertTo(matteIm, CV_32F, 255.0);
		    imwrite(matteImPath, matteIm);
            //imwrite(unshadImPath, unshadIm);
        }
	}
}
