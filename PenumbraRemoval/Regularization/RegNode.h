#ifndef REG_NODE_H
#define REG_NODE_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "RegGraphParams.h"
#include "RegLabel.h"

class RegNode {
public:
	RegNode();
  explicit RegNode(int id);
  //RegNode(const RegNode& other);
	
  ~RegNode() {};

  //// DEBUG ONLY: see if adding white patch improves results
  //void AddWhiteLabel() {
  //  EigenMat p(labels_[0].patch().size());
  //  p.setConstant(1.0f);
  //  labels_.push_back(RegLabel(p));
  //};

  void RemoveParentLink() { parent_ = NULL; }

	int id() const { return id_; }

	RegNode* parent() { return parent_; }
	void set_parent(RegNode* parent, const ParentOffset& parent_offset) {
		parent_ = parent;
    parent_offset_ = parent_offset;
	}

  ParentOffset parent_offset() const { return parent_offset_; }

	RegNode* peer(int index) { return peers_[index]; }
	void set_peer(int index, RegNode* peer) { peers_[index] = peer; }

	float unary_cost(int label) const { return costs_[label]; }
	void set_unary_cost(int label, float cost) { costs_[label] = cost; }
  void add_unary_cost(float cost) { costs_.push_back(cost); }

	void add_child(RegNode* child) { children_.push_back(child); }

	int n_labels() const { return int(labels_.size()); }

	RegLabel* label(int p) { return &labels_[p]; }
	void set_labels(
      const std::vector<EigenMat>& patches,
      const std::vector<float>& costs);
  void AddLabel(const RegLabel& label) { labels_.push_back(label); }

	bool IsConnected() const;

  static bool IsUnder(int coarse_level, int coarse_x, int coarse_y, 
                      int fine_level,   int fine_x,   int fine_y);
    
  // DBG
  std::vector<float> variances_;
private:
  void Init();

	int id_;
	RegNode* parent_;
	std::vector<RegNode*> peers_;		// stored in the order N, E, S, W
	std::vector<RegNode*> children_;	// stored in row-wise order starting from top-left
  ParentOffset parent_offset_; // two integers representing pixel offset within
                               // the parent to get the corresponding subpatch

	std::vector<RegLabel> labels_;
	std::vector<float> costs_;	// unary costs for all the patches
};

#endif // REG_NODE_H