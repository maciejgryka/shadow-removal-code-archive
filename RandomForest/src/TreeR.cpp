#include "TreeR.h"
#include "NodeR.h"
#include <DataProvider/Serializer.h>


#include <iostream>
//#include <ofstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
//#include <limits>

using namespace RandomForest;

double log2(double n)  {  
	if(n == 0) {
		return(-100000000.0); // hack
	} else {
		return log( n ) / log( 2.0 );
	}
}

NodeIdType getDepthFromId(NodeIdType id) {
	return NodeIdType(log2(id+1.1));
};

TreeR::TreeR(
    int id,
    int max_depth,
    int n_dim_in,
    int n_dim_out,
    int min_sample_count):

    id_(id),
    depth_(0),
    max_depth_(max_depth),
    n_nodes_(0),
    n_leaves_(0),
    n_dim_in_(n_dim_in),
    n_dim_out_(n_dim_out),
    min_sample_count_(min_sample_count) {}

TreeR::TreeR(
    int id,
    int depth,
    int n_nodes,
    int n_leaves,
    int n_dim_in,
    int n_dim_out):

    id_(id),
	  depth_(depth),
    max_depth_(depth),
	  n_nodes_(n_nodes),
	  n_leaves_(n_leaves),
	  n_dim_in_(n_dim_in),
	  n_dim_out_(n_dim_out),
    min_sample_count_(-1) {}

TreeR::~TreeR() {
	delete root_;
}

void TreeR::AddNodeToTree(
    NodeIdType id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity) {
  NodeR* node_to_add = 
      new NodeR(id, is_leaf, split_dim, split_thresh, impurity);
	bool added = AddNodeToTree(node_to_add);
  // if the node was not added to the tree, something went wrong
  // we have to delete it
  if (!added) { 
    std::cerr << "ERROR: Node with id '" << id << "' couldn't be added to the tree!" << std::endl;
    delete node_to_add; 
  }
}

void TreeR::AddNodeToTree(
    NodeIdType id,
    bool is_leaf,
    int split_dim,
    double split_thresh,
    double impurity,
    const NormalDistribution& nd,
    const std::vector<int>& samples) {
	AddNodeToTree(new NodeR(id, is_leaf, split_dim, split_thresh, impurity, nd, samples));
}

bool TreeR::AddNodeToTree(NodeR* node_to_add) {
	if (node_to_add->id() == 0 && nodes_.find(0) == nodes_.end()) {
		root_ = node_to_add;
    nodes_[0] = node_to_add;
    return true;
	} else {
    NodeIdType parent_id(node_to_add->GetParentId());
    // if the parent ID does not exist in nodes_, we cannot add the node that was passed in
    if (nodes_.find(parent_id) == nodes_.end()) {
      return false;
    }
    // if the parent does exist, connect the two
    NodeR* parent(nodes_.at(parent_id));
    // the parent cannot be a leaf
    if (parent->is_leaf()) {
      return false;
    }
    node_to_add->set_parent(parent);
    // node_to_add is a left child if its ID is odd, right child if even
    if (node_to_add->id() % 2 != 0) {
      parent->set_left(node_to_add);
    } else {
      parent->set_right(node_to_add);
    }
    nodes_[node_to_add->id()] = node_to_add;
	  return true;
  }
}

void TreeR::CreateChild(
    NodeR* parent,
    ChildType child_type,
    const EigenMatCM& data,
    const EigenMat& labels,
    const std::vector<int>& samples) {
	// create (left or right) child node
	NodeIdType newId(parent->id() * 2 + 1 + int(child_type));
	NodeR* child = new NodeR(newId, data, labels, samples);
	child->set_parent(parent);
	if (child_type == CHILD_LEFT) {
		parent->set_left(child);
	} else {
		parent->set_right(child);
	}
	
	++n_nodes_;
	
  // if this node doesn't have enough examples to produce two children
  // or if it's already at maximum depth, make it a leaf
  if (child->n_samples() < 2*min_sample_count_ || getDepthFromId(newId) >= max_depth_) {
		SetNodeToLeaf(child);
	}
}

void TreeR::OptimizeNode(
    NodeR* node,
    int n_dim_trials,
    int n_thresh_trials,
    const EigenMatCM& data,
    const EigenMat& labels) {
	int node_depth = getDepthFromId(node->id());
	if (node_depth > depth_) { depth_ = node_depth; }
	if (depth_ == max_depth_) {
		SetNodeToLeaf(node);
		return;
	} else if (depth_ > max_depth_) {
		std::cerr << "Error: maxDepth exceeded." << std::endl;
	}

	// initialise tests - currently not taking care of duplicates
  std::vector<int> test_id(randomIntVector(n_dim_trials, 0, n_dim_in_));
	
	// case where the dimensionality is small - exhaustive search
	if (n_dim_trials >= n_dim_in_) { 
		n_dim_trials = n_dim_in_;
    test_id = std::vector<int>(n_dim_trials);
		for (int idIt = 0; idIt < n_dim_in_; ++idIt) {
			test_id[idIt] = idIt;
		}
	}

	double dMin;
	double dMax;
	// set up random threshold test values
	EigenMat test_thresh(n_dim_trials, n_thresh_trials);
	for (int idIt =0; idIt < n_dim_trials; idIt ++) {
		dMin = node->dim_min(test_id[idIt]);
		dMax = node->dim_max(test_id[idIt]);
		for (int thIt =0; thIt < n_thresh_trials; thIt ++) { 
			test_thresh(idIt, thIt) = float(dMin + (dMax - dMin) * ((double)rand() / (double)RAND_MAX));
		}
	}

  // evaluate information gain for each proposed split and save best one
	double infoGainBest = -DBL_MAX;
	int bestId = 0;
	float bestThresh = 0;
  // this is a float, because we're using it for modifying the information gain
  float n_samples = static_cast<float>(node->n_samples());
	std::vector<int> bestSamplesL;
	std::vector<int> bestSamplesR;

	bool foundSomething = false;

  // DBG
  std::vector<float> info_gains(n_dim_trials * n_thresh_trials, -1.0f);

  double impurity_left;
  double impurity_right;
	for (int idIt = 0; idIt < n_dim_trials; ++idIt) { // loop through possible dims
		for (int thIt = 0; thIt < n_thresh_trials; ++thIt) { // try different threshes at this dimension
			std::vector<int> samplesL;
			std::vector<int> samplesR;
      samplesL.reserve(node->n_samples());
      samplesR.reserve(node->n_samples());

			// loop through valid training points
      for (std::vector<int>::const_iterator it = node->samples().begin(); it != node->samples().end(); ++it) {
 				if (data(*it, test_id[idIt]) < test_thresh(idIt, thIt)) {
					samplesL.push_back(*it);
				} else {
					samplesR.push_back(*it);
				}
			}

			// don't bother evaluating if we don't have enough somples at each node
      if (samplesL.size() < min_sample_count_ || samplesR.size() < min_sample_count_) { 
        continue;
      }
			// compare information gain resulting from this split to best one so far
			impurity_left  = Entropy(labels, samplesL) * static_cast<float>(samplesL.size()) / n_samples;
			impurity_right = Entropy(labels, samplesR) * static_cast<float>(samplesR.size()) / n_samples;

			double infoGain = node->impurity() - (impurity_left + impurity_right);
      info_gains[idIt * n_thresh_trials + thIt] = infoGain;
			if (infoGain > infoGainBest) {
				foundSomething = true;
				bestId = test_id[idIt];
				bestThresh = test_thresh(idIt, thIt);
				infoGainBest = infoGain;
				bestSamplesL = samplesL;
				bestSamplesR = samplesR;
			}
		}
	}

	// if no split produces better results than status quo, don't split
  // TODO is this allowed to happen in a proper tree?
	if (!foundSomething) {
		SetNodeToLeaf(node);
		return;
	}

	node->set_split_dim(bestId);
	node->set_split_thresh(bestThresh);
  // DBG
  node->info_gains_ = info_gains;

	CreateChild(node, CHILD_LEFT, data, labels, bestSamplesL);
	CreateChild(node, CHILD_RIGHT, data, labels, bestSamplesR);

	if (!node->left()->is_leaf()) {
		OptimizeNode(node->left(), n_dim_trials, n_thresh_trials, data, labels);
	} 

	if (!node->right()->is_leaf()) {
		OptimizeNode(node->right(), n_dim_trials, n_thresh_trials, data, labels);
	}
}

// note: this is deliberately not inlined to avoid including NodeR.h in the header
NormalDistribution TreeR::Test(const RowVector& x) {
	return root_->EvaluatePoint(x);
};

void TreeR::Train(
    const EigenMatCM& data,
    const EigenMat& labels,
    const std::vector<int>& bag_samples,
    int n_dim_trials,
    int n_thresh_trials) {
	root_ = new NodeR(0, data, labels, bag_samples);
	++n_nodes_;
	OptimizeNode(root_, n_dim_trials, n_thresh_trials, data, labels);
}

NodeR* TreeR::GetTerminalNode(const RowVector& features) {
  return root_->GetTerminalNode(features);
}

void TreeR::SetNodeToLeaf(NodeR* node) {
  //std::cout << node->impurity() << ", ";
	node->set_to_leaf();
	++n_leaves_;
}

void TreeR::WriteTree(std::ofstream& out_stream) const {
	out_stream << depth_ << std::endl;
	out_stream << n_nodes_ << std::endl;
	out_stream << n_leaves_ << std::endl;

	root_->WriteNode(out_stream);
}

void TreeR::WriteTreeBin(std::ofstream& out_stream) const {
  StreamWriter sw(&out_stream);
  sw.WriteInt(depth_);
  sw.WriteInt(n_nodes_);
  sw.WriteInt(n_leaves_);

  root_->WriteNodeBin(out_stream);
}

void TreeR::ReadTreeBin(std::ifstream& in_stream) {
  StreamReader sr(&in_stream);
  depth_ = sr.ReadInt();
  n_nodes_ = sr.ReadInt();
  n_leaves_ = sr.ReadInt();
  //std::cout << "depth " << depth_ << std::endl;
  //std::cout << "n_nodes " << n_nodes_ << std::endl;
  //std::cout << "n_leaves " << n_leaves_ << std::endl;

  root_ = new NodeR();
  root_->ReadNodeBin(in_stream);
  nodes_[0] = root_;

  for (int n = 0; n < n_nodes_ - 1; ++n) {
    NodeR* curr_node = new NodeR();
    curr_node->ReadNodeBin(in_stream);
    AddNodeToTree(curr_node);
  }
}


std::vector<int> RandomForest::randomIntVector(int size, int min, int max) {
  std::vector<int> vec(size);
  for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    *it = min + rand() % (max-min);
  }
  return vec;
}