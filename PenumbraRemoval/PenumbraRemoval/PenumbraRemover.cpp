#include "PenumbraRemover.h"
#include <ctime>
#include <list>
#include <algorithm>
#include <iterator>
#include <DataProvider/DataProvider.h>
#include <DataProvider/Serializer.h>
#include <Regularization/RegNode.h>
#include <Regularization/PlaneFitter.h>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

// DBG
#include <TreeR.h>
#include <NodeR.h>

using namespace std;
using namespace Eigen;
using namespace cv;

using namespace RandomForest;

const double PenumbraRemover::kCharToFloatCoeff = 1.0/255.0;
const double PenumbraRemover::kFloatToCharCoeff = 255.0;

inline bool comparator(const PenumbraRemover::FloatInt& lhs, const PenumbraRemover::FloatInt& rhs) {
  return lhs.first < rhs.first;
}

//rotate the given matrix 90deg clock-wise
void Rot90(EigenMat* mat) {
  assert(mat->rows() == mat->cols());
  int m(mat->rows());

  // permutation matrix
  EigenMat perm(m, m);
  perm.setZero();
  for (int r(0); r < m; ++r) {
    perm(r, m - 1 - r) = 1.f;
  }

  // rotate 90 degrees
  *mat = mat->transpose() * perm;
}

PenumbraRemover::PenumbraRemover(int finest_scale, int n_scales, int scale_step, float unary_cost_scaling):
	finest_scale_(finest_scale),
	n_scales_(n_scales),
  scale_step_(scale_step),
  unary_cost_scaling_(unary_cost_scaling),
  uniform_finest_(false)
{
}

PenumbraRemover::~PenumbraRemover(void) {
}

std::list<Rect> PenumbraRemover::GetPatchCoords(const Mat& im, int scale) {
	assert(im.channels() == 1);

	int w(im.size().width);
	int h(im.size().height);

	int patch_side(int(pow(2.0, double(scale))));

	int n_patches_hor(int(w / patch_side));
	int n_patches_vert(int(h / patch_side));

	std::list<Rect> patch_coords;
	for (int pv = 0; pv < n_patches_vert; ++pv) {
		for (int ph = 0; ph < n_patches_hor; ++ph) {
      patch_coords.push_back(Rect(ph*patch_side, pv*patch_side, patch_side, patch_side));
		}
	}
	return patch_coords;
}

void PenumbraRemover::RemoveUnwantedPatches(
    std::list<Rect>& patch_coords,
    const Mat& mask_im,
    std::vector<int>* masked_patches) {
	assert(mask_im.channels() == 1);
  // prepare masked_patches vector
	*masked_patches = vector<int>(patch_coords.size());
	int counter = 0;
	// iterate through patches in the list and remove ones that don't include
  // any masked pixels or that are too close to bounadry
  for (list<Rect>::iterator it = patch_coords.begin(); it != patch_coords.end(); ++counter) {
    if (IsAllZeros(Mat(mask_im, *it))) {// || !DataProvider::IsInImage(DataProvider::ExpandRect(*it, PATCH_FEATURE_EXTENSION), mask_im)) {
			it = patch_coords.erase(it);
			(*masked_patches)[counter] = 0;
		} else {
			++it;
			(*masked_patches)[counter] = 1;
		}
	}
}

// trains the forest at one scale
void PenumbraRemover::TrainScale(
    int scale,
		const EigenMat& data,
		const EigenMat& labels,
    const Options& options,
		const std::string& ensemble_file,
		const std::string& pcaw_file,
    bool compute_pca,
    const std::string& leaves_path) {
  if (compute_pca) {
	  // create PCA object and save result
    std::cout << "computing PCA params" << std::endl;
    pcaws_[scale] = PCAWrapper(labels, options.GetParamInt("n_dim_out"), pcaw_file);
  } else {
    std::cout << "reading PCA params" << std::endl;
    pcaws_[scale].deserialize(pcaw_file);
  }

  EigenMat projected_labels(pcaws_[scale].Project(labels));

  //// DBG display eigenvectors
  //for (int ev(0); ev < 4; ++ev) {
  //  Mat evec(pcaws_[scale].eigenvector(ev));
  //  evec = evec.reshape(1, 32);
  //  normalize(evec, evec, 0.0, 1.0, CV_MINMAX);
  //  imshow("evec", evec);
  //  waitKey();
  //  destroyAllWindows();
  //}

  //std::cout << "PRE:" << std::endl;
  //std::cout << "RowMajor: " << (data.Flags & Eigen::RowMajorBit) << std::endl;
  //std::cout << data.block(0, 0, 1, 5) << std::endl << std::endl;
  //std::cout << data.data() << " " << data.data()[0] << " " << data.data()[1] << " " << data.data()[2] << " " << data.data()[3] << std::endl;

  clock_t t0(clock());

	// train RF at this scale
	forests_[scale] = Ensemble(
      options.GetParamInt("n_trees"),
			options.GetParamInt("tree_depth"),
      data.cols(),
      options.GetParamInt("n_dim_out"));
	forests_[scale].Train(
      data,
			projected_labels,
			static_cast<int>(data.cols() * options.GetParamFloat("fraction_dim_trials")),
			options.GetParamInt("n_thresh_trials"),
			options.GetParamFloat("bag_prob"),
			options.GetParamInt("min_sample_count"));
	is_trained_[scale] = true;
  //forests_[scale].WriteEnsemble(ensemble_file);
  forests_[scale].WriteEnsembleBin(ensemble_file);

  std::cout << "FOREST TRAINING TIME: " << static_cast<double>(clock() - t0)/CLOCKS_PER_SEC << std::endl;

  //std::cout << "POST:" << std::endl;
  //std::cout << "RowMajor: " << (data.Flags & Eigen::RowMajorBit) << std::endl;
  //std::cout << data.block(0, 0, 1, 5) << std::endl << std::endl;
  //std::cout << data.data() << " " << data.data()[0] << " " << data.data()[1] << " " << data.data()[2] << " " << data.data()[3] << std::endl;


  //// DBG test on the training set
  //EigenMat sample_leaf_impurities(data.rows(), forests_[scale].n_trees());
  //// for each sample in the training set
  //for (int r = 0; r < data.rows(); ++r) {
  //  // test it using each tree and get the impurity of the node where it lands
  //  vector<NodeR*> leaf_nodes = forests_[scale].GetAllTerminalNodes(data.row(r));
  //  // save the result in a matrix, where each row is a trainig point and each
  //  //  column is leaf node impurity from a given tree
  //  for (int c = 0; c < leaf_nodes.size(); ++c) {
  //    sample_leaf_impurities(r, c) = leaf_nodes[c]->impurity();
  //  }
  //}
  //// serialize the resulting matrix
  //DataProvider::serializeEigenMatAscii(sample_leaf_impurities, "C:\\Work\\VS2010\\PenumbraRemoval\\x64\\data\\sample_leaf_impurities.csv", false);

  //// DBG
  //if (!leaves_path.empty()) {
  //  int patch_size = DataProvider::getPatchSize(scale);
  //  // iterate over trees
  //  vector<TreeR*> trees = forests_[scale].trees();
  //  typedef vector<TreeR*>::iterator TVIter;
  //  // write only the first 5 trees
  //  int n_trees_to_write = 2;
  //  for (TVIter it = trees.begin(); it != trees.begin()+n_trees_to_write/*trees.end()*/; ++it) {
  //    stringstream tss;
  //    tss << (*it)->id();
  //    string tree_dir = leaves_path + tss.str() + "\\";
  //    // write out leaf nodes
  //    (*it)->root()->WriteNodeLabels(data, projected_labels, tree_dir, string("label_"), pcaws_[scale], patch_size);
  //  }
  //}
}

void PenumbraRemover::TestProjection(
    int scale,
	  const EigenMat& labels,
	  const std::string& pca_file,
	  const std::string& out_folder) {
	pcaws_[scale] = PCAWrapper(pca_file);
  EigenMat projected_labels = pcaws_[scale].Project(labels);
  EigenMat back_projected_labels = pcaws_[scale].BackProject(projected_labels);
  DataProvider::serializeEigenMatAscii(projected_labels, out_folder + "projected_labels.csv");
  int patch_size = DataProvider::getPatchSize(scale);
  for (int tl = 0; tl < back_projected_labels.rows(); ++tl) {
		EigenMat patch(back_projected_labels.row(tl));
		patch.resize(patch_size, patch_size);
		Mat patch_cv;
		eigen2cv(patch, patch_cv);
		patch_cv.convertTo(patch_cv, CV_32F, 255.0);
		char num[10];
		sprintf_s(num, "%d", tl);
		bool wrote = imwrite(out_folder + "patch" + std::string(num) + ".png", patch_cv);
	}
}

EigenMat replicate(RowVectorXf row, int n_rows) {
	EigenMat out(n_rows, row.cols());
	--n_rows;
	for (; n_rows>= 0; --n_rows) {
		out.row(n_rows) = row;
	}
	return out;
}

Mat PenumbraRemover::Test(
        const Mat& shad_im,
        const Mat& mask_im,
        const Mat& gmatte_im,
        const Mat& unshad_mask_im,
        const Options& options) {
    //// DBG for writing out offset histograms
    //vector<RotatedRect> alignments;
    //// DBG for drawing offsets or visualization
    //Mat drawing_board(shad_im);
    //cvtColor(drawing_board, drawing_board, CV_GRAY2BGR);

    //get number of channels
    int n_channels = shad_im.channels();

    Mat guessed_matte_cv;
    if (unary_cost_ == 5) {
        // create a fist-guess approximation for the matte
        Mat shad_char;
        shad_im.convertTo(shad_char, CV_8UC1, kFloatToCharCoeff);
        Mat mask_char;
        mask_im.convertTo(mask_char, CV_8UC1, kFloatToCharCoeff);
        Mat unshad_mask_char;
        unshad_mask_im.convertTo(unshad_mask_char, CV_8UC1, kFloatToCharCoeff);

        int margin(options.GetParamInt("margin"));

        cv::Size original_size(options.GetParamInt("original_width"), options.GetParamInt("original_height"));
        Rect crop_region(margin, margin, original_size.width, original_size.height);

        if (options.GetParamInt("plane_inpaint") == 1) {
            guessed_matte_cv = GuessMatte(shad_char, mask_char, unshad_mask_char, options);
            Mat gunshadp;
            cv::divide(shad_im, guessed_matte_cv, gunshadp);
            cv::imwrite(
            options.GetParamString("image_folder") + "\\" + options.GetParamString("image_name") + "_gunshadp.png",
            gunshadp(crop_region) *255);
        } else {
            guessed_matte_cv = gmatte_im;
            Mat gmatte_im_corpped(gmatte_im(crop_region));
            cv::imwrite(options.GetParamString("results_dir") + "\\" + options.GetParamString("image_name") + "_gmatte.png", gmatte_im_corpped*255);
        }
        //// DBG
        //if (options.IsParamValid("grid_offset") && options.GetParamBool("grid_offset")) {
        //  assert(options.IsParamValid("grid_offset_x") && options.IsParamValid("grid_offset_y"));
        //  const int patch_size(16);
        //  // make the image one patch_size smaller in each dimension and start cutting it at (new_x, new_y) offset
        //  int new_width(guessed_matte_cv.cols - patch_size);
        //  int new_height(guessed_matte_cv.rows - patch_size);

        //  int new_x(options.GetParamInt("grid_offset_x"));
        //  int new_y(options.GetParamInt("grid_offset_y"));
        //
        //  //cv::imshow("original", guessed_matte_cv(cv::Rect(0, 0, new_width, new_height)));
        //  guessed_matte_cv = guessed_matte_cv(cv::Rect(new_x, new_y, new_width, new_height));
        //  //cv::imshow("offset", guessed_matte_cv);
        //  //cv::waitKey();
        //  //cv::destroyAllWindows();
        //}
    }
    EigenMat guessed_matte;
    DataProvider::CvToEigen(guessed_matte_cv, guessed_matte);

    // split images into separate channels for processing
    vector<Mat> shad_im_channels(n_channels);
    split(shad_im, shad_im_channels);
    vector<Mat> mask_im_channels(n_channels);
    split(mask_im, mask_im_channels);

    Mat original_mask(mask_im_channels[0]);
    // dilate each mask channel
    for (auto img_ch(mask_im_channels.begin()); img_ch != mask_im_channels.end(); ++img_ch) {
        dilate(*img_ch, *img_ch, cv::Mat(), cv::Point(-1, -1), 25);
    }

    // create vector of output images (one per channel)
    vector<Mat> out_im_channels(n_channels);

    // create vector of empty regularization graphs (one per channel)
    vector<RegGraph*> graphs;
    for (int ch = 0; ch < n_channels; ++ch) {
        //graphs.push_back(new RegGraph(n_scales_, finest_scale_, scale_step_, shad_im.cols - PATCH_MARGIN*2, shad_im.rows - PATCH_MARGIN*2));
        graphs.push_back(new RegGraph(n_scales_, finest_scale_, scale_step_, shad_im.cols, shad_im.rows));
    }

    std::map<int, vector<int> > masked_patches;
    // initialize DataProvider object with the right features
    DataProvider dp;
    dp.set_active_features(active_features_);
    // prepare a set of constant patches and corresponding (uniform) unary costs
    // to put at each node at the finest scale
    vector<EigenMat> constant_patches;
    int finest_patch_size = DataProvider::getPatchSize(finest_scale_);
    constant_patches.reserve(N_CONSTANT_PATCHES);
    for (int p = 0; p < N_CONSTANT_PATCHES+1; ++p) {
        constant_patches.push_back(EigenMat(finest_patch_size, finest_patch_size));
        constant_patches[p].setConstant(1.0f - p * 1.f/N_CONSTANT_PATCHES);
    }
    vector<float> constant_unary_costs(constant_patches.size(), 1.0f);

    // for each scale
    for (int s = finest_scale_; s <finest_scale_ + (n_scales_ * scale_step_); s += scale_step_) {
        // make sure that both Ensemble and PCAWrapper exist at this scale
        // either in memory or serialized
        // unless this is the finest scale, in which case don't load anything - it's all uniform patches
        // TODO: is this right?
        if (!(s == finest_scale_ && uniform_finest_) && !is_trained_[s]) {
            string ensemble_file(options.GetParamString("ensemble_file"));
            string pca_file(options.GetParamString("pca_file"));

            if (ensemble_file.empty() || pca_file.empty()) {
                cerr << "regressor not trained and no data files given" << endl;
                throw;
            } else {
                cout << "reading saved forest at scale " << s << "...";
                ensemble_file = DataProvider::insertScaleNum(ensemble_file, s);
                pca_file = DataProvider::insertScaleNum(pca_file, s);
                pcaws_[s] = PCAWrapper(pca_file);
                forests_[s].ReadEnsembleBin(ensemble_file);
                is_trained_[s] = true;
                cout << "done" << endl;
            }
        }
        // if we want to use unary cost type 1 (Mahalanobis distance in feature space)
        // we need to load feature data
        EigenMat data;
        if (unary_cost_ == 1 && s != finest_scale_) {
            char dataPath[PATH_LENGTH];
            sprintf(dataPath, data_file_.c_str(), s);
            cout << "reading data at scale " << s << " ...";
            data = DataProvider::deserializeEigenMat<float>(dataPath);
            cout << " done" << endl;
        }
        // create a matrix of labels in PCA space
        EigenMat projected_labels(labels_matrix_[s]->rows(), forests_[s].n_dim_out());
        projected_labels = pcaws_[s].Project(*labels_matrix_[s]);

        int patch_size(DataProvider::getPatchSize(s));
        // for each channel
        for (int ch = 0; ch < n_channels; ++ch) {
            // initialize output to 1.0
            out_im_channels[ch] = Mat::ones(shad_im.size(), CV_32F);
            EigenMat outImChEigen;
            DataProvider::CvToEigen(out_im_channels[ch], outImChEigen);

            // divide shadow image into patches
            list<Rect> patchCoords = GetPatchCoords(shad_im_channels[ch], s);
            RemoveUnwantedPatches(patchCoords, mask_im_channels[ch], &(masked_patches[s]));
            // connect all the nodes at this level
            graphs[ch]->ConnectLevel(s, masked_patches[s]);
            // if previous level exists, connect the current level to it
            if (s > finest_scale_) {
                graphs[ch]->ConnectLevels(s, masked_patches[s], s - scale_step_, masked_patches[s - scale_step_]);
            }

            // for each of the patches
            for (list<Rect>::iterator it = patchCoords.begin(); it != patchCoords.end(); ++it) {
                // if this is the finest scale, assign nTrees uniform labels spanning the space 1..255
                if (s == finest_scale_ && uniform_finest_) {
                    RegNode* currNode = graphs[ch]->GetNodeAtPixel(s, (*it).x, (*it).y);
                    currNode->set_labels(constant_patches, constant_unary_costs);
                    continue;
                }
                EigenMat labels;
                vector<EigenMat> patches;
                vector<float> unaryCosts;
                // create feature vector from the patch
                Scalar original_rect_color(0.0, 0.0, 255.0);
                Scalar aligned_rect_color(0.0, 255.0, 0.0);

                AlignedPatch aligned_patch(cv::RotatedRect(cv::Point(it->x + patch_size/2, it->y + patch_size/2), cv::Size(patch_size, patch_size), 0.f), 0.f);

                // make a copy of the original position to be used for "unalignemnt" later, once we have the label
                const cv::RotatedRect initial_rect(aligned_patch.rot_rect());

                // align the rect and compute the offset
                DataProvider::Align(shad_im, options.GetParamBool("align_rot"), options.GetParamInt("align_trans"), &aligned_patch);

                const RotatedRectOffset original_offset(
                    aligned_patch.rot_rect().center - initial_rect.center,
                    aligned_patch.rot_rect().angle - initial_rect.angle,
                    aligned_patch.intensity_offset());

                // expand to compute the features over a larger window
                DataProvider::ExpandRotRect(PATCH_FEATURE_EXTENSION, aligned_patch.rot_rect_ptr());

                //// DBG vizualization
                //Mat board(shad_im);
                //cvtColor(board, board, CV_GRAY2BGR);
                //DataProvider::DrawRotatedRect(initial_rect, Scalar(0.0, 0.0, 255.0), &board);
                //DataProvider::DrawRotatedRect(aligned_patch.rot_rect(), Scalar(0.0, 255.0, 0.0), &board);

                //RotatedRect unaligned(aligned_patch.rot_rect());
                //unaligned.center.x -= original_offset.offset().x;
                //unaligned.center.y -= original_offset.offset().y;
                //unaligned.angle -= original_offset.angle();
                //DataProvider::ShrinkRotRect(PATCH_FEATURE_EXTENSION, &unaligned);

                //std::cout << original_offset.offset().x << ", " << original_offset.offset().y << ", " << original_offset.angle() << std::endl;


                //Mat cutout_int(DataProvider::CutOutRotatedRect(shad_im, aligned_patch.rot_rect()));
                //imshow("feature", cutout_int);

                //DataProvider::DrawRotatedRect(unaligned, Scalar(255.0, 0.0, 0.0), &board);
                //imshow("board", board);

                RowVectorXf features(dp.GetFeatureVectorEigen(shad_im_channels[ch], mask_im, guessed_matte_cv, aligned_patch));

                //imshow("board", board);
                //imshow("after", DataProvider::Unalign(expanded, original_offset));

                //imshow("expanded", expanded);

                //waitKey();
                //destroyAllWindows();

                // get indices of samples in chosen leaf nodes
                vector<int> all_samples = forests_[s].TestGetAllSamples(features);
                // make sure we don't have duplicates
                std::unique(all_samples.begin(), all_samples.end());

                for (int samp(0); samp < all_samples.size(); ++samp) {
                    EigenMat temp_patch(labels_matrix_.at(s)->row(all_samples[samp]));
                    temp_patch.resize(patch_size*PATCH_FEATURE_EXTENSION, patch_size*PATCH_FEATURE_EXTENSION);
                    //temp_patch.transposeInPlace();
                    cv::Mat temp_patch_cv;
                    eigen2cv(temp_patch, temp_patch_cv);
                    RotatedRect unaligned2(aligned_patch.rot_rect());

                    int n_intensities(64);
                    float increment(2.f/n_intensities);
                    RotatedRectOffset temp_offset(original_offset);
                    temp_offset.set_intensity_offset(0.f);
                    Mat sub_patch(DataProvider::Unalign(temp_patch_cv, temp_offset, &unaligned2));
                    for (int ii(0); ii < n_intensities + 1; ++ii) {
                        //std::cout << temp_offset.intensity_offset() << " ";
                        Mat temp_sub_patch(sub_patch - 1.f + ii * increment);

                        DataProvider::CvToEigen(temp_sub_patch, temp_patch);
                        patches.push_back(temp_patch);
                        //imshow("sub", sub_patch);
                        //waitKey();
                    }
                }
                if (unary_cost_ == 2 || unary_cost_ == 4) {
                    // create a matrix of labels, where each row is a sample from a leaf
                    // node, except the first row, which is the mean of them all
                    labels = MatrixXf(all_samples.size()+1, forests_[s].n_dim_out());
                    labels.row(0).setZero();
                    for (int samp = 1; samp < all_samples.size()+1; ++samp) {
                        labels.row(samp) = projected_labels.row(all_samples[samp-1]);
                        labels.row(0) += labels.row(samp);
                    }
                    labels.row(0) /= all_samples.size();
                }
                // backproject into high-dim space using PCA
                // after this each row is a flattened patch
                //labels = pcaws_[s].BackProject(labels);
                // reshape each row into a patch
                //patches.reserve(labels.rows());
                //for (int r = 0; r < labels.rows(); ++r) {
                //    patches.push_back(matrixxf(labels.row(r)));
                //    patches[r].resize(patchsize, patchsize);
                //    patches[r].transposeinplace();
                //}

                // calculate unary cost
                if (unary_cost_ == 0) {
                    unaryCosts = vector<float>(patches.size(), 1.0f);
                } else if (unary_cost_ == 1) {
                    // unaryCosts = forests_[s].GetUnaryCosts(features, data);
                } else if (unary_cost_ == 2) {
                      unaryCosts = forests_[s].GetUnaryCostsFromMeanLabel(labels);
                      unaryCosts.erase(unaryCosts.begin());
                } else if (unary_cost_ == 3) {
                      std::vector<NodeR*> terminal_nodes = forests_[s].GetAllTerminalNodes(features);
                      unaryCosts = forests_[s].GetUnaryCostsFromImpurity(terminal_nodes);
                } else if (unary_cost_ == 4) {
                      std::vector<NodeR*> terminal_nodes = forests_[s].GetAllTerminalNodes(features);
                      unaryCosts = forests_[s].GetUnaryCostsFromMeanAndImpurity(labels, terminal_nodes);
                } else if (unary_cost_ == 5) {
                      EigenMat gmatte_im_eigen;
                      DataProvider::CvToEigen(guessed_matte_cv, gmatte_im_eigen);
                      unaryCosts = forests_[s].GetUnaryCostsFromGuessedMatte(patches, gmatte_im_eigen, *it);
                }

                // now that we have all the labels, we'll sort them according to unary costs and only keep top N
                KeepTopNLabels(options.GetParamInt("keep_top_n_labels"), &patches, &unaryCosts);
                // save as candidates (and their associated costs) at a given node in the graph
                RegNode* currNode = graphs[ch]->GetNodeAtPixel(s, (*it).x, (*it).y);
                //currNode->variances_ = variances;
                currNode->set_labels(patches, unaryCosts);
            }
        }
    }

    // clean the graph (remove unconnected nodes)
    for (int ch = 0; ch < n_channels; ++ch) {
        std::cout << "collapsing the graph" << std::endl;
        // collapse multiscale graph into one with finest-scale nodes that include
        // sliced up labels from all higher scales
        graphs[ch]->Collapse(unary_cost_scaling_);
        std::cout << "cleaning the graph" << std::endl;
        graphs[ch]->Clean(options.GetParamBool("set_bound"));
    }
    std::cout << "regularizing... ";
    std::vector<EigenMat> out;
    std::vector<EigenMat> out_variances(n_channels);
    out.reserve(n_channels);
    // regularize the graphs
    for (int ch = 0; ch < n_channels; ++ch) {
        out.push_back(graphs[ch]->Regularize());
        std::cout << "finished regularizing channel " << ch << std::endl;
        eigen2cv(out[ch], out_im_channels[ch]);

        cv::Mat matte_fill(cv::Mat::ones(shad_im.rows, shad_im.cols, CV_32F));
        cv::multiply(matte_fill, 1.0f - original_mask, matte_fill);
        cv::multiply(out_im_channels[ch], mask_im_channels[ch], out_im_channels[ch]);
        out_im_channels[ch] += matte_fill;
        std::cout << "converted to cv::Mat" << std::endl;
        //// normalize out_variances
        //out_variances[ch] += EigenMat::Ones(out_variances[ch].rows(), out_variances[ch].cols()) * -out_variances[ch].minCoeff();
        //out_variances[ch] /= out_variances[ch].maxCoeff();
    }
    std::cout << "done" << std::endl;

    // create output image
    Mat outIm;
    merge(out_im_channels, outIm);

    //// DBG for writing out all found offsets
    //string rot_rects_path("C:\\Work\\research\\shadow_removal\\experiments\\output\\size2\\rot_rects\\");
    //std::ofstream out_offsets;
    //out_offsets.open(rot_rects_path + "offsets.txt");
    //out_offsets << "angles = [";
    //for (int rr(0); rr < alignments.size(); ++rr) {
    //  out_offsets << alignments[rr].angle << ",";
    //}
    //out_offsets << "]" << std::endl << "x = [";
    //for (int rr(0); rr < alignments.size(); ++rr) {
    //  out_offsets << alignments[rr].center.x << ",";
    //}
    //out_offsets << "]" << std::endl << "y = [";
    //for (int rr(0); rr < alignments.size(); ++rr) {
    //  out_offsets << alignments[rr].center.y << ",";
    //}
    //out_offsets << "]" << std::endl;
    //out_offsets.close();

    //imshow("drawing_board", drawing_board);
    //waitKey();
    //destroyAllWindows();

    return outIm;
}

double Log2(double n) {
  // return log(n) / log(2.0);
  return log(n) / 0.6931471805599453;
}

float NormalizeAndInvert(float val, float normalization) {
  return 1.f - val/normalization;
}

float PenumbraRemover::GetNdcg(
      const std::vector<FloatInt>& dists_gt,
      const std::vector<FloatInt>& unary_costs_sorted,
      int p) {
  const float kDiscountBase = 1.1f;
  float max_dist = dists_gt.back().first;
  // relevance is inversely proportional to the distance
  float ideal_dcg = NormalizeAndInvert(dists_gt[0].first, max_dist);
  // get index of the next element in unary_costs_sorted
  int sorted_index = unary_costs_sorted[0].second;
  // find the corresponding distance in dists_gt
  float dcg = NormalizeAndInvert(find_if(
      dists_gt.begin(),
      dists_gt.end(),
      [sorted_index](const FloatInt& fi) {
        return fi.second == sorted_index;
      })->first, max_dist);

  float discount;
  for (int el = 1; el < p; ++el) {
    discount = pow(kDiscountBase, static_cast<float>(el)); //Log2(static_cast<double>(el+1));
    ideal_dcg += NormalizeAndInvert(dists_gt[el].first, max_dist) / discount;
    sorted_index = unary_costs_sorted[el].second;
    dcg += NormalizeAndInvert(find_if(
      dists_gt.begin(),
      dists_gt.end(),
      [sorted_index](const FloatInt& fi){return fi.second==sorted_index;})->first,
      max_dist) / discount;
  }
  return dcg/ideal_dcg;
}

// create a guessed matte from shadow image and mask
cv::Mat PenumbraRemover::GuessMatte(
    const cv::Mat& shad,
    const cv::Mat& mask,
    const cv::Mat& unshad_mask,
    const Options& options) {

  assert(shad.type() == CV_8UC1 && shad.channels() == 1);
  assert(mask.type() == CV_8UC1 && mask.channels() == 1);
  assert(unshad_mask.type() == CV_8UC1 && unshad_mask.channels() == 1);

  const int kBlurKernelSize = 0;
  const int kBlurSigma = 5;

  Mat shad_local(shad.clone()); // local copy of the shad agument (we need to convert it to float)
  shad_local.convertTo(shad_local, CV_32F, kCharToFloatCoeff);
  // blur the input if required
  if (kBlurKernelSize > 0) {
    cv::GaussianBlur(shad_local, shad_local, cv::Size(kBlurKernelSize, kBlurKernelSize), kBlurSigma);
  }

  typedef enum {kFlat = 0, kLeastSquares, kRansac} FitType;

  //const FitType kFitType(FitType::kFlat);
  //const FitType kFitType(FitType::kLeastSquares);
  const FitType kFitType(FitType::kRansac);

  Mat unshad_mask_local(unshad_mask);
  unshad_mask_local.convertTo(unshad_mask_local, CV_32F, kCharToFloatCoeff);

  Mat fit_to; // pixels to fit the plane to
  multiply(shad_local, unshad_mask_local, fit_to);

  // convert back to 0-255 range
  fit_to.convertTo(fit_to, CV_8UC1, kFloatToCharCoeff);
  unshad_mask_local.convertTo(unshad_mask_local, CV_8UC1, kFloatToCharCoeff);

  // extract all the points in the unshad_mask_local with [x, y, z] coordinates, where
  // z comes from corresponding coordinate in shad_local
  vector<Point2i> non_zero_nm_points;
  GetNonZeroPoints(unshad_mask_local, &non_zero_nm_points);

  // replace the masked region in the blurred input with the plane
  Mat noshad_guess(shad_local.clone());
  vector<Point2i> non_zero_m_points;
  GetNonZeroPoints(mask, &non_zero_m_points);

  // fit a plane to the extracted points
  Vec4f coeffs;
  switch (kFitType) {
  case kFlat:
  {
    // fitting a flat plane is equivalent (I think) to filling the rest with
    // the mean out-of-shadow value
    float mean_oos_val(cv::mean(fit_to, unshad_mask_local)[0] / 255.f);

    // DBG: add a constant bias to the guess
    mean_oos_val += 0.02f;

    for (vector<Point2i>::const_iterator point = non_zero_m_points.begin();
        point != non_zero_m_points.end();
        ++point) {
      noshad_guess.at<float>(*point) = mean_oos_val;
    }
    break;
  }
  case kRansac:
  {
    // using RANSAC
    Mat plane_points(non_zero_nm_points.size(), 3, CV_32F);
    Mat temp_row(1, 3, CV_32F);
    int row_index = 0;
    for (auto point = non_zero_nm_points.begin();
        point != non_zero_nm_points.end();
        ++point, ++row_index) {
      temp_row.at<float>(0) = static_cast<float>(point->x);
      temp_row.at<float>(1) = static_cast<float>(point->y);
      temp_row.at<float>(2) = static_cast<float>(fit_to.at<unsigned char>(*point));
      temp_row.copyTo(plane_points.row(row_index));
    }
    EigenMat plane_points_eigen;
    DataProvider::CvToEigen(plane_points, plane_points_eigen);
    Eigen::Vector4f plane(PlaneFitter::GetBestPlaneRansac(plane_points_eigen, 10000, 8.f));
    coeffs = Vec4f(plane.x(), plane.y(), plane.z(), plane.w());
    for (auto point = non_zero_m_points.begin();
        point != non_zero_m_points.end();
        ++point) {
      float val((coeffs(0)*point->x  + coeffs(1)*point->y + coeffs(3)) / -coeffs(2));
      val /= 255.f; // convert to 0-1 range
      noshad_guess.at<float>(*point) = val;
    }
    break;
  }
  case kLeastSquares:
  {
    // using least squres
    Mat plane_points(non_zero_nm_points.size(), 3+1, CV_32F);
    Mat temp_row(1, 3+1, CV_32F);
    temp_row.at<float>(3) = 1.0f;
    int row_index = 0;
    for (auto point = non_zero_nm_points.begin();
        point != non_zero_nm_points.end();
        ++point) {
      // also pad the last column of the point matrix with zeros
      temp_row.at<float>(0) = static_cast<float>(point->x);
      temp_row.at<float>(1) = static_cast<float>(point->y);
      temp_row.at<float>(2) = static_cast<float>(fit_to.at<unsigned char>(*point));
      temp_row.copyTo(plane_points.row(row_index));
      ++row_index;
    }
    cv::SVD svd;
    Mat w, u, vt;
    svd.compute(plane_points, w, u, vt);

    //std::cout << vt << std::endl;

    coeffs = vt.row(3);
    for (vector<Point2i>::const_iterator point = non_zero_m_points.begin();
        point != non_zero_m_points.end();
        ++point) {
      float val = (coeffs(0)*point->x  + coeffs(1)*point->y + coeffs(3)) / -coeffs(2);
      val /= 255.f; // convert to 0-1 range
      noshad_guess.at<float>(*point) = val;
    }
    break;
  }
  }

  // divide the blurred input by the guessed unshadowed image to obtain a matte guess
  Mat guessed_matte;
  cv::divide(shad_local, noshad_guess, guessed_matte);
   // make sure that the estimated matte is never larger than 1.0
   cv::threshold(guessed_matte, guessed_matte, 1.0, 1.0, THRESH_TRUNC);
  //// normalize the matte to make sure no value is larger than 1.0
  //double matte_max;
  //minMaxLoc(guessed_matte, NULL, &matte_max);
  //guessed_matte /= matte_max;
  //// also ensure that only masked pixels are < 1.0
  //guessed_matte.setTo(1.0, 255 - mask);

  Mat guessed_result(shad.clone());
  guessed_result.convertTo(guessed_result, CV_32F, 1.0/255.0);
  divide(guessed_result, guessed_matte, guessed_result);

  //imshow("fit_to",          fit_to.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //imshow("shad",            shad.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //imshow("shad_local",      shad_local.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //imshow("noshad_guess",    noshad_guess.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //imshow("guessed_matte",   guessed_matte.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //imshow("guessed_result",  guessed_result.colRange(PATCH_MARGIN, shad.cols-PATCH_MARGIN).rowRange(PATCH_MARGIN, shad.rows-PATCH_MARGIN));
  //waitKey();
  //destroyAllWindows();

  return guessed_matte;
}

void PenumbraRemover::SetMarginsTo(int margin, float val, cv::Mat* img) {
  // calculate size of the original image (without margins)
  cv::Size inner_size(img->cols - 2*margin, img->rows - 2*margin);
  img->rowRange(0, margin) = val;
  img->rowRange(margin + inner_size.height, img->rows) = val;

  img->colRange(0, margin) = val;
  img->colRange(margin + inner_size.width, img->cols) = val;
}


void PenumbraRemover::GetNonZeroPoints(const cv::Mat& img, std::vector<Point2i>* non_zero_points) {
  assert(img.channels() == 1);
  assert(img.type() == CV_8UC1);
  assert(non_zero_points->empty());

  non_zero_points->reserve(img.rows * img.cols);
  for (int r = 0; r < img.rows; ++r) {
    for (int c = 0; c < img.cols; ++c) {
      // std::cout << r << ", " << c << " = " << static_cast<int>(img.at<unsigned char>(r, c)) << std::endl;
      if (img.at<unsigned char>(r, c) > 0) {
        non_zero_points->push_back(Point2i(c, r));
      }
    }
  }
  non_zero_points->shrink_to_fit();
}

void PenumbraRemover::KeepTopNLabels(
    int n,
    std::vector<EigenMat>* labels,
    std::vector<float>* unary_costs) {
  if (n > labels->size()) {
    return;
  }
  // sort the unary costs
  vector<FloatInt> unary_costs_sorted;
  for (int uc(0); uc < unary_costs->size(); ++uc) {
    unary_costs_sorted.push_back(FloatInt(unary_costs->at(uc), uc));
  }
  sort(unary_costs_sorted.begin(), unary_costs_sorted.end(), comparator);

  // keep only the top n
  std::vector<EigenMat> new_labels;
  new_labels.reserve(n);
  std::vector<float> new_unary_costs;
  new_unary_costs.reserve(n);

  for (int l(0); l < n; ++l) {
    new_labels.push_back(labels->at(unary_costs_sorted[l].second));
    new_unary_costs.push_back(unary_costs->at(unary_costs_sorted[l].second));
  }

  *labels = new_labels;
  *unary_costs = new_unary_costs;
}