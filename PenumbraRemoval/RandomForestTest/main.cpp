#include <ctime>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <DataProvider/DataProvider.h>
#include <DataProvider/PCAWrapper.h>
#include <RandomForest/Ensemble.h>

using namespace cv;


const int nScales = 1;

const std::string dataName("patches");

const std::string dataPath("C:\\Work\\VS2010\\PenumbraRemoval\\RandomForestTest\\data\\");
const std::string ensembleFile(dataPath + dataName + "_trees.data");

void train() {
	int nTrees = 50;
	int treeDepth = 20;
	int nDimTrials = 40;
	int nThreshTrials = 200;
	float bagProb = 0.66;
	int minExsAtNode = 5;

	Matrix2df data(DataProvider::deserializeMatrix2df(dataPath + dataName + "_x.csv"));
	Matrix2df labels(DataProvider::deserializeMatrix2df(dataPath + dataName + "_y.csv"));

	Ensemble trees;
	int t = clock();
	std::cout << std::endl << "Training..." << std::endl;	
	trees.setParams(nTrees, treeDepth, data.cols(), labels.cols());		
	trees.train(data, labels, nDimTrials, nThreshTrials, bagProb, minExsAtNode);
	std::cout << "\r                                      " << std::endl;
	std::cout << "training time " << double(clock() - t) / double(CLOCKS_PER_SEC) << " sec" << std::endl << std::endl;

	// save forest
	trees.writeEnsemble(ensembleFile);
}

void test() {
	Ensemble trees;
	trees.loadEnsemble(ensembleFile);

	Matrix2df dataTest(DataProvider::deserializeMatrix2df(dataPath + dataName + "_x_test.csv"));
	Matrix2df labelsTest(dataTest.rows(), trees.getNDimOut());
	
	int t = clock();
	std::cout << std::endl << "Testing..." << std::endl;	
	for (int r = 0; r < dataTest.rows(); ++r) {
		labelsTest.getEigenMat().row(r) = trees.test(dataTest.getEigenMat().row(r));
	}
	std::cout << "testing time " << double(clock() - t) / double(CLOCKS_PER_SEC) << " sec" << std::endl << std::endl;

	std::string opFileName(dataPath + dataName + "_y_test.csv");
	DataProvider::serializeMatrix2df(labelsTest, opFileName);
}

void main(int argc, char* argv[]) {
	train();
	test();
}