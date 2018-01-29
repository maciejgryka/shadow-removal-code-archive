#ifndef REG_LABEL_H
#define REG_LABEL_H

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "RegGraphParams.h"
#include "PlaneFitter.h"
//typedef Eigen::Vector2f NormalType;

class RegLabel {
public:
	RegLabel():
		  patch_(0,0),
		  hasSubLabels_(false) {};

	explicit RegLabel(const EigenMat& patch):
		  patch_(patch),
		  hasSubLabels_(false) {
		if (patch.rows() > 1 && patch.cols() > 1) {
			hasSubLabels_ = true;
		}
	};

	~RegLabel() {};

  RegLabel SubLabel(int xCoord, int yCoord, int size) const {
    return RegLabel(patch_.block(yCoord, xCoord, size, size));
  }

  RegLabel SubLabel(ParentOffset offset, int size) const {
    return SubLabel(offset.first, offset.second, size);
  }

	// calculate how compatible the two patches are to be put next to each other in the graph 
	// as specified by relationship
	static float LabelDistance(const RegLabel& l1, const RegLabel& l2, EdgeType relationship);

  int LabelSize() const {
    return int(patch_.cols());
  };

  static int LabelSize(int level) {
    return int(pow(2.0, level));
  };

  static void setWeights(float relationship_peer, float relationship_parent, float beta) {
    relationship_weight_peer_ = relationship_peer;
    relationship_weight_parent_ = relationship_parent;
    beta_weight_ = beta;
  };

  const EigenMat& patch() const { return patch_; };

  static float relationship_weight_peer_;
  static float relationship_weight_parent_;
  static float beta_weight_;

private:
	float CalcMeanIntensity(EigenMat patch) { return patch.sum()/patch.count(); };

  EigenMat patch_;
	bool hasSubLabels_;
};

#endif // REG_LABEL_H