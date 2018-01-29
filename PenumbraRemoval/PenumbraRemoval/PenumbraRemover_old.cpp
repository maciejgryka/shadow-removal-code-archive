#include "PenumbraRemover.h"
#include <math.h>
#include <DataProvider/DataProvider.h>

void reconstruct_image(Mat shad_im, Mat graph) {
//	regularize the graph
//	replace appropriate regions in the input image with the graph output
//	
//	return unshadowed image
}

PenumbraRemover::PenumbraRemover(int nScales, const EnsembleParams& ensembleParams):
	mNScales(nScales),
	mPcaws(nScales),
	mForests(nScales),
	mEp(ensembleParams),
	mTrained(false)
{
}


PenumbraRemover::~PenumbraRemover(void) {
}

std::list<Rect> PenumbraRemover::getPatchCoords(const Mat& im, int scale) {
	assert(im.channels() == 1);

	int w(im.size().width);
	int h(im.size().height);

	int patchSide(int(pow(2.0, double(scale))));

	int nPatchesHor(int(w / patchSide));
	int nPatchesVert(int(h / patchSide));

	std::list<Rect> patchCoords;
	for (int ph = 0; ph < nPatchesHor; ++ph) {
		for (int pv = 0; pv < nPatchesVert; ++pv) {
			patchCoords.push_back(Rect(ph*patchSide, pv*patchSide, patchSide, patchSide));
		}
	}
	return patchCoords;
}

void PenumbraRemover::removeUnmaskedPatches(std::list<Rect>& patchCoords, const Mat& maskIm) {
	assert(maskIm.channels() == 1);

	std::list<Rect>::iterator it;
	
	for (it = patchCoords.begin(); it != patchCoords.end();) {
		if (isAllZeros(Mat(maskIm, *it))) {
			it = patchCoords.erase(it);
		} else {
			++it;
		}
	}
}

// trains the forest at one scale
void PenumbraRemover::trainScale(int scaleId, 
								 const Matrix2df& data, 
								 const Matrix2df& labels, 
								 int nPcaDims, 
								 const std::string& ensembleFile,
								 const std::string& pcawFile) {
	// training is done once (NOT per channel) because for now we assume that:
	// 1. different color lights have the same fall-off characteristics so we can
	// treat them the same
	// 2. patches from all channels were included in the training set on equal rights

	// convert labels to OpenCV
	Mat labelsCv;
	eigen2cv(labels.getEigenMat(), labelsCv);
	
	// run PCA on labels
	mPcaws[scaleId] = PCAWrapper(labelsCv, nPcaDims);
	mPcaws[scaleId].serialize(pcawFile);

	// project labels into principle components space
	Mat projectedLabelsCv(mPcaws[scaleId].project(labelsCv));
	//testReprojection(projectedLabelsCv, scaleId);

	// convert back to Eigen
	MatrixXf projectedLabels;
	cv2eigen(projectedLabelsCv, projectedLabels);
	
	// train RF at this scale
	mForests[scaleId] = Ensemble(mEp.getNTrees(), mEp.getTreeDepth(), mEp.getNDimIn(), mEp.getNDimOut());
	mForests[scaleId].train(data, projectedLabels, mEp.getNDimTrials(), mEp.getNThreshTrials(), mEp.getBagProb(), mEp.getMinExsAtNode());
	mForests[scaleId].writeEnsemble(ensembleFile);

	//writeTrainingSetOutput(scaleId, data, projectedLabels);

	mTrained = true;
}

void PenumbraRemover::writeTrainingSetOutput(int scaleId,
											 const Matrix2df& data,
											 const std::string& ensembleFile,
											 const std::string& pcaFile,
											 const std::string& outFile) 
{
	
	// load trained forest and PCA coeffs							  
	mForests[scaleId].loadEnsemble(ensembleFile);
	mPcaws[scaleId] = PCAWrapper(pcaFile);

	int patchSize = DataProvider::getPatchSize(scaleId);

	// regression
	MatrixXf output(data.rows(), mForests[scaleId].getNDimOut());
	for (int r = 0; r < data.rows(); ++r) {
		output.row(r) = mForests[scaleId].test(data.row(r));

		MatrixXf treeLabelsPCA = mForests[scaleId].testGetAll(data.row(r));
		DataProvider::serializeMatrix2df(Matrix2df(treeLabelsPCA), "C:\\Work\\VS2010\\PenumbraRemoval\\x64\\data\\tree_patches\\labelsPCA.csv");
		MatrixXf treeLabels = mPcaws[scaleId].backProject(treeLabelsPCA);

		//cout << label << endl;

		for (int tl = 0; tl < treeLabels.rows(); ++tl) {
			MatrixXf patch(treeLabels.row(tl));
			patch.resize(patchSize, patchSize);
			Mat patchCv;
			eigen2cv(patch, patchCv);
			patchCv.convertTo(patchCv, CV_32F, 255.0);
			char num[10];
			sprintf_s(num, "%d", tl);
			bool wrote = imwrite("C:\\Work\\VS2010\\PenumbraRemoval\\x64\\data\\tree_patches\\patch" + std::string(num) + ".png", patchCv);
		}
	}

	// back-project from PCA space and write to file
	output = mPcaws[scaleId].backProject(output);
	DataProvider::serializeMatrix2df(Matrix2df(output), outFile);
}

Mat PenumbraRemover::test(int scaleId, const RowVectorXf& features) {
	RowVectorXf labels(mForests[scaleId].test(features));
	Mat labelsCv;
	eigen2cv(labels, labelsCv);
	Mat patch(mPcaws[scaleId].backProject(labelsCv));

	int patchSize(DataProvider::getPatchSize(scaleId));

	patch = patch.reshape(0,patchSize);

	return patch;
}

Mat PenumbraRemover::test(const Mat& shadIm, const Mat& maskIm, const string& ensembleFile, const string& pcawFile) {
	//assert(mTrained);

	//get number of channels
	int nChannels = shadIm.channels();

	// split images into separate channels for processing
	std::vector<Mat> shadImCh(nChannels);
	std::vector<Mat> maskImCh(nChannels);
		
	split(shadIm, shadImCh);
	split(maskIm, maskImCh);

	// create vector of output images (one per channel)
	std::vector<Mat> outImCh(nChannels);

	// create nChannels empty regularization graphs

	
	// for each scale
	// TODO: remove scale-dependence
	for (int sc = 5; sc < 6/*mNScales*/; ++sc) {
		// make sure that both Ensemble and PCAWrapper exist at this scale
		// either in memory or serialized
		if (!mTrained) {
			if (ensembleFile.empty() || pcawFile.empty()) {
				cerr << "regressor not trained and no data files passed" << endl;
				throw;
			} else {
				cout << "reading saved forest... ";
				mPcaws[sc] = PCAWrapper::deserialize(pcawFile);
				mForests[sc].loadEnsemble(ensembleFile);
				mTrained = true;
				cout << "done" << endl;
			}
		}
		// for each channel
		for (int ch = 0; ch < nChannels; ++ch) {
			// initialize output to 1.0
			outImCh[ch] = Mat::ones(shadIm.size(), CV_32F);

			MatrixXf outImChEigen;
			cv2eigen(outImCh[ch], outImChEigen);

			int patchSize(DataProvider::getPatchSize(sc));

			// divide shadow image into patches
			std::list<Rect> patchCoords = getPatchCoords(shadImCh[ch], sc);
			
			// take only patches, which include mask pixels
			removeUnmaskedPatches(patchCoords, maskImCh[ch]);

			std::list<Rect>::iterator it;
			//std::sha
			// for each of the patches
			for (it = patchCoords.begin(); it != patchCoords.end(); ++it) {
				// create feature vector from the patch
				Mat patch(shadImCh[ch], *it);
				RowVectorXf features = DataProvider::getFeatureVectorEigen(patch);

				// TODO: get k low-dim representations from the RF
				// regress label
				RowVectorXf label = mForests[sc].test(features);

				// backproject using PCA
				MatrixXf patchMatte = mPcaws[sc].backProject(label);

				// reshape into a patch
				patchMatte.resize(patchSize, patchSize);
				patchMatte.transposeInPlace();

				// get ROI from the output image and assign the output to it
				outImChEigen.block((*it).y, (*it).x, (*it).height, (*it).width) = patchMatte;
				eigen2cv(outImChEigen, outImCh[ch]);

				// save as k candidates at a given node in the graph
				// save unary costs info per candidate in the graph
			}
		}
	}

	// regularize the graph

	// create output image
	Mat outIm;
	merge(outImCh, outIm);

	return outIm;
}

bool PenumbraRemover::isAllZeros(Mat im) {
	im.convertTo(im, CV_32F);
	for (int x = 0; x < im.size().width; ++x) {
		for (int y = 0; y < im.size().height; ++y) {
			if (im.at<float>(y, x) > 0) {
				return false;
			}
		}
	}
	return true;
}