#include "RegLabel.h"
#include <Eigen/Core>
#include <Eigen/Dense>

float RegLabel::relationship_weight_peer_;
float RegLabel::relationship_weight_parent_;
float RegLabel::beta_weight_;

float RegLabel::LabelDistance(const RegLabel& label1, const RegLabel& label2, EdgeType relationship) {
 	float intensityDistance(0.0f);
  float relationship_weight;

  bool isPeer = (relationship == EDGE_X ||  relationship == EDGE_Y);
 	if (isPeer) {
    relationship_weight = relationship_weight_peer_;
 		Eigen::VectorXf p1Intensity, p2Intensity;
 		if (relationship == EDGE_X) {
 			// retrieve appropriate columns from intensity patches
 			p1Intensity = label1.patch().col(label1.patch().cols()-1);
 			p2Intensity = label2.patch().col(0);
 		} else if (relationship == EDGE_Y) {
 			// retrieve appropriate rows from intensity patches
 			p1Intensity = label1.patch().row(label1.patch().rows()-1).transpose();
 			p2Intensity = label2.patch().row(0).transpose();
		}
    //intensityDistance = (p2Intensity - p1Intensity).cwiseProduct(p2Intensity - p1Intensity).sum();
    intensityDistance = (p2Intensity - p1Intensity).norm();
    intensityDistance = intensityDistance < 0.1f ? 0.0f : intensityDistance;
 	} else {
    relationship_weight = relationship_weight_parent_;
    //intensityDistance = (label1.patch() - label2.patch()).norm();
    intensityDistance = (label1.patch() - label2.patch()).cwiseAbs().sum() / label1.patch().count();
 	}
   return relationship_weight * beta_weight_ * intensityDistance;
 }
