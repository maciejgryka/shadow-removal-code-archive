#include "RegNode.h"
//#include "RegLabel.h"

RegNode::RegNode():
    id_(-1),
	  parent_(NULL) {
  Init();
}

RegNode::RegNode(int id):
    id_(id),
    parent_(NULL) {
  Init();
}

//RegNode::RegNode(const RegNode& other):
//    id_(other.id_),
//    peers_(other.peers_),
//    children_(other.children_),
//    parent_offset_(other.parent_offset_),
//    labels_(other.labels_),
//    costs_(other.costs_) {}

void RegNode::Init() {
  peers_.reserve(N_PEERS);
  for (int p = 0; p < N_PEERS; ++p) {
		peers_.push_back(NULL);
  }

  children_.reserve(N_CHILDREN);
  for (int c = 0; c < N_CHILDREN; ++c) {
		children_.push_back(NULL);
  }
}

void RegNode::set_labels(const std::vector<EigenMat>& patches, const std::vector<float>& costs) {
	assert(patches.size() == costs.size());

  if (!labels_.empty()) { labels_.clear(); }
  labels_.reserve(patches.size());

  std::vector<EigenMat>::const_iterator it;
  for (it = patches.begin(); it != patches.end(); ++it) {
    EigenMat patch = *it;
		labels_.push_back(RegLabel(patch));
	}
	costs_ = costs;
};

bool RegNode::IsConnected() const {
	if (parent_ != NULL) { return true; }
		
	for (int p = 0; p < N_PEERS; ++p) {
		if (peers_[p] != NULL) { return true; }
	}

  for (int c = 0; c < N_CHILDREN; ++c) {
    if (children_[c] != NULL) { return true; }
  }

	return false;
}

bool RegNode::IsUnder(
    int coarse_level, int coarse_x, int coarse_y,
    int fine_level, int fine_x, int fine_y) {
  // (square root of) the number of pixel per label at both levels
  int coarse_size(RegLabel::LabelSize(coarse_level));
  int fine_size(RegLabel::LabelSize(fine_level));

  // (sqare root of) the number of fine nodes corresponding to one coarse node
  int size_difference = (coarse_size / fine_size);

  // calculate node coordinates on the fine level that fall under the coarse node
  // min is inclusive, max is exclusive
  int min_fine_x = coarse_x * size_difference;
  int min_fine_y = coarse_y * size_difference;
  
  int max_fine_x = (coarse_x + 1) * size_difference;
  int max_fine_y = (coarse_y + 1) * size_difference;

  return fine_x >= min_fine_x && fine_x < max_fine_x && 
         fine_y >= min_fine_y && fine_y < max_fine_y;
}