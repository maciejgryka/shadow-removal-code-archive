#include "RegGraph.h"
#include "RegLabel.h"

using namespace std;
using namespace Eigen;

RegGraph::RegGraph():
        n_nodes_(-1),
        n_connected_nodes_(-1),
        nodes_(0),
        connected_nodes_(0) {}

RegGraph::RegGraph(int n_scales, int finest_scale, int scale_step, int finest_width, int finest_height):
        n_scales_(n_scales),
        finest_scale_(finest_scale),
        scale_step_(scale_step),
        width_(finest_width),
        height_(finest_height),
        n_nodes_(0),
        n_connected_nodes_(0),
        connected_nodes_(NULL) {
    // ensure finest scale dimensions are divisible by the coarsest scale
    int coarsest_size =  RegLabel::LabelSize(finest_scale  + (n_scales-1) * scale_step);
    if (!IsDivisibleBy(finest_width, coarsest_size) || !IsDivisibleBy(finest_height, coarsest_size)) {
        std::cerr << "Image width and height have to be divisible by the coarsest-scale patch size!" << endl;
        std::cerr << "coarsest scale: " << coarsest_size << "; image size: " << finest_width << "x" << finest_height << endl;
        throw;
    }

    // calculate overall number of nodes in the graph
    for (int s = finest_scale_ ; s < finest_scale_ + n_scales_ * scale_step_; s += scale_step_) {
        n_level_nodes_[s] = GetNNodesAtLevel(s);
        n_nodes_ += n_level_nodes_[s];
    }

    nodes_.reserve(n_nodes_);
    for (int n = 0; n < n_nodes_; ++n) {
        nodes_.push_back(RegNode(n));
    }
}

int RegGraph::GetNNodesAtLevel(int level) {
  if (level == 0) {
    return width_ * height_;
  } else {
    return (width_ / RegLabel::LabelSize(level)) * (height_ / RegLabel::LabelSize(level));
  }
}

int RegGraph::GetNNodesUpTo(int level) const {
	int sum(0);
	level -= scale_step_;
	for (; level >= finest_scale_; level -= scale_step_) {
		sum += n_level_nodes_.at(level);
	}
	return sum;
}

void RegGraph::Clean(bool set_bound) {
	int nConnectedNodes(0);
	vector<bool> connected(n_nodes_, false);
	for (int n = 0; n < n_nodes_; ++n) {
		if (nodes_[n].IsConnected()) {
			if (nodes_[n].n_labels() == 0) {
				std::cerr << "Connected node without patches!" << std::endl;
			}
			// optionally set boundary conditions
			if (set_bound) {
				bool isFinestScale = nodes_[n].id() < GetNNodesAtLevel(finest_scale_);
				bool isOnEdge = HasUnconnectedNeighbors(nodes_[n]);
				// if this node is at the finest level and at the edge, set it to constant white
				if (isOnEdge) {
				SetUnconnectedNeighborsToConstant(&nodes_[n], 1.0f);
				}
				connected[n] = true;
				++nConnectedNodes;
			}
		}
	}
	connected_nodes_.reserve(nConnectedNodes);
	for (int n = 0; n < n_nodes_; ++n) {
		if (connected[n]) {
			connected_nodes_.push_back(&(nodes_[n]));
		}
	}
}

void RegGraph::Collapse(float unary_cost_scaling) {
  // get number of finest-scale nodes
  int n_nodes_finest = GetNNodesAtLevel(finest_scale_);
  int finest_patch_size = RegLabel::LabelSize(finest_scale_);
  // for each fine node
  for (int n = 0; n < n_nodes_finest; ++n) {
    RegNode* fine_node = GetNode(n);
    // skip if teh node is not connected to anything
    if (!fine_node->IsConnected()) continue;
    // get pixel coordinates of <fine_node>
    int fine_pixel_x = GetPixelCoordX(*fine_node);
    int fine_pixel_y = GetPixelCoordY(*fine_node);
    // for each coarser scale
    for (int s = finest_scale_ + scale_step_;
        s < finest_scale_ + (n_scales_ * scale_step_);
        s += scale_step_) {
      // find the coarse node that this fine node lies under
      RegNode* coarse_node = GetNodeAtPixel(s, fine_pixel_x, fine_pixel_y);
      // make sure that the coarse node is connected
      assert(coarse_node->IsConnected());
      // get pixel coordinates of <coarse_node>
      int coarse_pixel_x = GetPixelCoordX(*coarse_node);
      int coarse_pixel_y = GetPixelCoordY(*coarse_node);
      // get ParentOffset of <fine_node> with respect to <coarse_node>
      ParentOffset po(fine_pixel_x - coarse_pixel_x, fine_pixel_y - coarse_pixel_y);
      // for each label of the coarse node
      for (int l = 0; l < coarse_node->n_labels(); ++l) {
        // cut out the sublabel patch and assign it to the fine node
        fine_node->AddLabel(coarse_node->label(l)->SubLabel(po, finest_patch_size));
        // TODO: this is a hack to weight coarse scales more
        fine_node->add_unary_cost(coarse_node->unary_cost(l) * pow(unary_cost_scaling, s - finest_scale_));
      }
    }
  }
  // delete all nodes ecept the finest scale
  RemoveCoarseScales();
}


void RegGraph::SetToConstant(RegNode* node, float value) {
  int level = GetNodeLevel(*node);
  // size of the patch at the level of the node
  int patch_size = RegLabel::LabelSize(level);
  EigenMat constant_patch(patch_size, patch_size);
  constant_patch.setConstant(value);
  
  vector<EigenMat> labels(1, constant_patch);
  
  vector<float> costs(1, 0.0f);
  node->set_labels(labels, costs);
  node->variances_.push_back(0.0f);
};

bool RegGraph::HasUnconnectedNeighbors(const RegNode& node) {
  vector<int> peer_ids;
  peer_ids.reserve(N_PEERS);
  peer_ids.push_back(GetNorthPeerId(node));
  peer_ids.push_back(GetEastPeerId(node));
  peer_ids.push_back(GetSouthPeerId(node));
  peer_ids.push_back(GetWestPeerId(node));

  for (vector<int>::const_iterator it = peer_ids.begin(); it != peer_ids.end(); ++it) {
    if (*it != -1) {
      RegNode* neighbor = &(nodes_[*it]);
      if (!neighbor->IsConnected()) {
        return true;
      }
    }
  }

  return false;
}

void RegGraph::SetUnconnectedNeighborsToConstant(RegNode* node, float value) {
  RegNode* neighbor;
  int neighbor_id;
  
  neighbor_id = GetNorthPeerId(*node);
  if (neighbor_id != -1) {
    neighbor = &(nodes_[neighbor_id]);
    if (!neighbor->IsConnected()) {
      SetToConstant(neighbor, value);
      AddEdge(neighbor, node, EDGE_Y);
    }
  }

  neighbor_id = GetEastPeerId(*node);
  if (neighbor_id != -1) {
    neighbor = &(nodes_[neighbor_id]);
    if (!neighbor->IsConnected()) {
      SetToConstant(neighbor, value);
      AddEdge(node, neighbor, EDGE_X);
    }
  }

  neighbor_id = GetSouthPeerId(*node);
  if (neighbor_id != -1) {
    neighbor = &(nodes_[neighbor_id]);
    if (!neighbor->IsConnected()) {
      SetToConstant(neighbor, value);
      AddEdge(node, neighbor, EDGE_Y);
    }
  }

  neighbor_id = GetWestPeerId(*node);
  if (neighbor_id != -1) {
    neighbor = &(nodes_[neighbor_id]);
    if (!neighbor->IsConnected()) {
      SetToConstant(neighbor, value);
      AddEdge(neighbor, node, EDGE_X);
    }
  }
}

void RegGraph::ConnectLevel(int level, const vector<int>& maskedNodes) {
	assert(maskedNodes.size() == n_level_nodes_[level]);

	int firstNodeIndex = (level == finest_scale_ ? 0 : GetNNodesUpTo(level));

  int patchSize = RegLabel::LabelSize(level);
	int levelWidth = width_ / patchSize;	// nodes per row at this level
	int levelHeight = height_ / patchSize;	// nodes per column at this level

	//RegNode* levelNodes(nodes_[firstNodeIndex]);
	for (int n = 0; n < n_level_nodes_[level]; ++n) {
		if (!maskedNodes[n]) {
			continue;
		}

		int x = n % levelWidth;
		int y = (n - x) / levelHeight;

		// connect to right neighbor if masked
		if (x < levelWidth - 1) {
			if (maskedNodes[n+1]) {
				AddEdge(&nodes_[firstNodeIndex+n], &nodes_[firstNodeIndex+n+1], EDGE_X);
			}
		}

		// connect to bottom neighbor if masked
		if (y < levelHeight - 1) {
			if (maskedNodes[n+levelWidth]) {
				AddEdge(&nodes_[firstNodeIndex+n], &nodes_[firstNodeIndex+n+levelWidth], EDGE_Y);
			}
		}
	}
}

void RegGraph::ConnectLevels(int coarseLevel,
							 const std::vector<int>& coarseMasked,
							 int fineLevel,
							 const std::vector<int>& fineMasked) 
{
	assert(coarseLevel > fineLevel);

	int firstNodeIndexF(fineLevel == finest_scale_ ? 0 : GetNNodesUpTo(fineLevel));
	int firstNodeIndexC(firstNodeIndexF); // index of the first node at coarse level
  for (int l = fineLevel; l < coarseLevel; ++l) {
    firstNodeIndexC += n_level_nodes_[l];
  }

  int coarsePatchSize(RegLabel::LabelSize(coarseLevel));  // size of patch at coarse level
	int coarseWidth(width_ / coarsePatchSize);		// nodes per row at coarse level
	int coarseHeight(height_ / coarsePatchSize);	// nodes per column at coarse level

	int finePatchSize(int(pow(2.0, fineLevel)));
	int fineWidth(width_ / finePatchSize);
	int fineHeight(width_ / finePatchSize);

  // for all nodes at coarser level
	for (int cn = 0; cn < n_level_nodes_[coarseLevel]; ++cn) {
		if (!coarseMasked[cn]) {  // skip if the node cn is unmasked 
			continue;
		}
    RegNode* parent(&(nodes_[firstNodeIndexC+cn]));
    int coarse_x(cn % coarseWidth); // coarse node coordinates
		int coarse_y((cn - coarse_x) / coarseHeight);
    
    // for all nodes at finer level
    for (int fn = 0; fn < n_level_nodes_[fineLevel]; ++fn) {
      int fine_x(fn % fineWidth);			// fine node coordinates
		  int fine_y((fn - fine_x) / fineHeight);
      
      // skip if the fine node fn is unmasked or not under current coarse node cn
      if (!fineMasked[fn] || !RegNode::IsUnder(coarseLevel, coarse_x, coarse_y, fineLevel, fine_x, fine_y)) {
        continue;
      }

      // calculate pixel offset to get a correct sublabel from within the coarse node
      int pixel_offset_x = (fine_x * finePatchSize) % coarsePatchSize;
      int pixel_offset_y = (fine_y * finePatchSize) % coarsePatchSize;

      ParentOffset po;
      po.first = pixel_offset_x;
      po.second = pixel_offset_y;

      RegNode* child(&(nodes_[firstNodeIndexF+fn]));
      // save the offset in the child and connect the nodes
      AddEdge(parent, child, EDGE_Z, po);
    }
	}
}

RegNode* RegGraph::GetNode(int level, int x, int y) {
	int levelWidth = width_ / int(pow(2.0, level));	// nodes per row at this level
	int index = GetNNodesUpTo(level) + (y * levelWidth) + x;
	return &(nodes_[index]);
}

RegNode* RegGraph::GetNodeAtPixel(int level, int pixel_x, int pixel_y) {
	int size(RegLabel::LabelSize(level));
	return GetNode(level, pixel_x/size, pixel_y/size);
}

int RegGraph::GetPixelCoordX(const RegNode& node) const {
  int level = GetNodeLevel(node);
  int label_size = RegLabel::LabelSize(level);
  // get grid coords of <node> and mutliply bu <label_size> to get pixel coords
  return GetNodeColumn(node) * label_size;
}

int RegGraph::GetPixelCoordY(const RegNode& node) const {
  int level = GetNodeLevel(node);
  int label_size = RegLabel::LabelSize(level);
  // get grid coords of <node> and mutliply bu <label_size> to get pixel coords
  return GetNodeRow(node) * label_size;
}

void RegGraph::AddEdge(RegNode* n1, RegNode* n2, EdgeType edge_type, ParentOffset parent_offset) {
  switch (edge_type) {
	case EDGE_X:
		n1->set_peer(1, n2);
		n2->set_peer(3, n1);
		break;
	case EDGE_Y: 
		n1->set_peer(2, n2);
		n2->set_peer(0, n1);
		break;
	case EDGE_Z:
    if (parent_offset == ParentOffset(-1, -1)) {
      std::cerr << "ERROR: RegGraph::AddEdge: invalid parent_offset" << std::endl;
    }
		n2->set_parent(n1, parent_offset);
		n1->add_child(n2);
		break;
	}
}

EigenMat RegGraph::Regularize(EigenMat* out_variances) {
    //////////// Construct MRF //////////////////////////////////
    MRFEnergy<TypeGeneral>* mrf;
    MRFEnergy<TypeGeneral>::NodeId* nodes;
    MRFEnergy<TypeGeneral>::Options options;
    TypeGeneral::REAL energy, lowerBound;

    lowerBound = 0.0;

    mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
    nodes = new MRFEnergy<TypeGeneral>::NodeId[n_nodes_];

    ///////////// Add data term /////////////////////////////
    TypeGeneral::REAL* data_term;

    int n_nodes = 0; // number of nodes which are active

    for (int i = 0; i < n_nodes_; ++i) {
        //std::cout << i << std::endl;
        int nLabels = nodes_[i].n_labels();
        nLabels = nLabels > 0 ? nLabels : 1;
        data_term = new TypeGeneral::REAL[nLabels];
        for (int k = 0; k < nLabels; ++k) {
            if (nodes_[i].IsConnected()) {
                data_term[k] =  nodes_[i].unary_cost(k);
            } else {
                data_term[k] = 1.0;
            }
        }
        nodes[i] = mrf->AddNode(TypeGeneral::LocalSize(nLabels), TypeGeneral::NodeData(data_term));
        ++n_nodes;
        delete[] data_term;
    }

    ///////////// Add edge term /////////////////////////////
    TypeGeneral::REAL* edge_term(NULL);
    bool initialized(false);
    int current_alloc_size(0);
    int desired_alloc_size(0);
    for (int i = 0; i < n_nodes_; ++i) {
        RegNode* currNode = &(nodes_[i]);
        RegNode* neighbor;
        // add left peer connection
        neighbor = currNode->peer(1);
        if (neighbor != NULL) {
            desired_alloc_size = currNode->n_labels() * neighbor->n_labels();
            ReallocIfNeeded(&edge_term, &desired_alloc_size, &current_alloc_size, &initialized);
            GetEdgeMatrix(edge_term, currNode, neighbor, EDGE_X);
            mrf->AddEdge(nodes[currNode->id()], nodes[neighbor->id()], TypeGeneral::EdgeData(TypeGeneral::GENERAL, edge_term));
        }
        // add bottom peer connection
        neighbor = currNode->peer(2);
        if (neighbor != NULL) {
            desired_alloc_size = currNode->n_labels() * neighbor->n_labels();
            ReallocIfNeeded(&edge_term, &desired_alloc_size, &current_alloc_size, &initialized);
            GetEdgeMatrix(edge_term, currNode, neighbor, EDGE_Y);
            mrf->AddEdge(nodes[currNode->id()], nodes[neighbor->id()], TypeGeneral::EdgeData(TypeGeneral::GENERAL, edge_term));
        }
        // add parent connection
        // NOTE: reverse order of neighbor and current node - parents always passed first!
        neighbor = currNode->parent();
        if (neighbor != NULL) {
            desired_alloc_size = neighbor->n_labels() * currNode->n_labels();
            ReallocIfNeeded(&edge_term, &desired_alloc_size, &current_alloc_size, &initialized);
            GetEdgeMatrix(edge_term, neighbor, currNode, EDGE_Z);
            mrf->AddEdge(nodes[neighbor->id()], nodes[currNode->id()], TypeGeneral::EdgeData(TypeGeneral::GENERAL, edge_term));
        }
    }
    if (initialized) { delete[] edge_term; };

    /////////////////////// TRW-S algorithm //////////////////////
    options.m_iterMax = 25; // maximum number of iterations
    mrf->Minimize_TRW_S(options, lowerBound, energy);
    //mrf->Minimize_BP(options, energy);

    int* labelOutPtr = new int[n_nodes_];

    for (int k = 0; k < n_nodes_; ++k){
        labelOutPtr[k] = mrf->GetSolution(nodes[k]);
    }
    std::cout << "got solution" << std::endl;

    // create the output image using optimized labels at the finest scale
    EigenMat out(height_, width_);
    out.setOnes();

    int patchSize(int(pow(2.0, finest_scale_)));
    int levelWidth(width_ / patchSize);		// nodes per row at the finest level
    int levelHeight(height_ / patchSize);	// nodes per column at the finest level

    bool write_variances = out_variances != NULL;
    // initialize EigenMat object where out_variances points to and set it to 0
    if (write_variances) {
        *out_variances = out;
        out_variances->setZero();
    }
    // assign variances to leaf node impurity image
    for (int y = 0; y < levelHeight; ++y) {
        for (int x = 0; x < levelWidth; ++x) {
            RegNode* currNode(&(nodes_[y*levelWidth + x]));
            // if this node has a label
            if (currNode->n_labels() != 0) {
                // get the corresponding block of the output image and assign the 
                // optimal patch to it
                Eigen::Vector2i topLeft(x*patchSize, y*patchSize);
                out.block(topLeft(1), topLeft(0), patchSize, patchSize) = currNode->label(labelOutPtr[y*levelWidth + x])->patch();
                if (write_variances) {
                  out_variances->block(topLeft(1), topLeft(0), patchSize, patchSize).setConstant(currNode->variances_[labelOutPtr[y*levelWidth + x]]);
                }
            }
        }
    }
    std::cout << "deleting" << std::endl;
    delete mrf;
    delete[] nodes;
    delete[] labelOutPtr;

    return out;
}

void RegGraph::ReallocIfNeeded(
    TypeGeneral::REAL** edge_term,
    int* desired_alloc_size,
    int* current_alloc_size,
    bool* initialized) {
  // if the memory has already been initialized
  if (*initialized) {
    // if currently allocated size is the same as needed, don't do anything
    if (*desired_alloc_size == *current_alloc_size) {
      return;
    } else {
      // if we need to re-allocate, delete old memory first
      delete[] *edge_term;
    }
  }
  // allocate required memory and set state
  *edge_term = new TypeGeneral::REAL[*desired_alloc_size];
  *initialized = true;
  *current_alloc_size = *desired_alloc_size;
}

void RegGraph::GetEdgeMatrix(
    TypeGeneral::REAL* edge_term,
    RegNode* n1,
    RegNode* n2, 
    EdgeType relationship) {
	
  int nLabels1(n1->n_labels());
  int nLabels2(n2->n_labels());

  // increase the cost of label disagreement at the edges (to ensure smooth transition to no-shadow)
  //float edge_cost_multiplier = nLabels1 == 1 || nLabels2 == 1 ? 5.0f : 1.0f;
  float edge_cost_multiplier = 1.0f;

  // in case it's a parent-child relationship, get appropriate sub-label from the parent
  ParentOffset po = n2->parent_offset();
  int sub_size = n2->label(0)->LabelSize();

	for(int l1 = 0; l1 < nLabels1; ++l1) {
		for (int l2 = 0; l2 < nLabels2; ++l2) {
      if (relationship == EDGE_X || relationship == EDGE_Y) {
        edge_term[l1+ l2*nLabels1] = RegLabel::LabelDistance(*(n1->label(l1)), *(n2->label(l2)), relationship) * edge_cost_multiplier;
      } else {
        edge_term[l1+ l2*nLabels1] = RegLabel::LabelDistance(n1->label(l1)->SubLabel(po, sub_size), *(n2->label(l2)), relationship) * edge_cost_multiplier;
      }
		}
	}
}

int RegGraph::GetNodeLevel(const RegNode& node) const {
  int nodes_so_far(0);
  for (std::map<int, int>::const_iterator it = n_level_nodes_.begin(); it !=n_level_nodes_.end(); ++it) {
    if (node.id() < (*it).second + nodes_so_far) return (*it).first;
    nodes_so_far += (*it).second;
  }
  // invalid - id indicates that the node is at a level that doesn't exist
  return -1;
};

int RegGraph::GetNorthPeerId(const RegNode& node) {
  int row = GetNodeRow(node);
    
  // if this node is in the top row of the image, there is no north peer
  if (row == 0) { return -1; }
  
  int levelWidth = GetLevelWidth(node);
  return node.id() - levelWidth;
};

int RegGraph::GetEastPeerId(const RegNode& node) {
  int col = GetNodeColumn(node);
    
  // if this node is in the right column of the image, there is no east peer
  if (col == GetLevelWidth(node) - 1) { return -1; }
    
  return node.id() + 1;
}

int RegGraph::GetSouthPeerId(const RegNode& node) {
  int row = GetNodeRow(node);

  // if this node is in the bottom row of the image, there is no south peer
  if (row == GetLevelHeight(node) - 1) { return -1; }
  
  int levelWidth = GetLevelWidth(node);
  return node.id() + levelWidth;
}

int RegGraph::GetWestPeerId(const RegNode& node) {
  int col = GetNodeColumn(node);
    
  // if this node is in the left column of the image, there is no west peer
  if (col == 0) { return -1; }
    
  return node.id() - 1;
}

int RegGraph::GetNodeColumn(const RegNode& node) const {
  int levelWidth = GetLevelWidth(node); // nodes per row at this level
  return (node.id() - GetNNodesUpTo(GetNodeLevel(node))) % levelWidth;
}

int RegGraph::GetNodeRow(const RegNode& node) const {
  int levelWidth = GetLevelWidth(node); // nodes per row at this level
  return (node.id() - GetNNodesUpTo(GetNodeLevel(node)))/ levelWidth;;
}

int RegGraph::GetLevelWidth(const RegNode& node) const {
  int level = GetNodeLevel(node);
  int patchSize = RegLabel::LabelSize(level);
  return width_ / patchSize;
}

int RegGraph::GetLevelHeight(const RegNode& node) const {
  int level = GetNodeLevel(node);
  int patchSize = RegLabel::LabelSize(level);
  return height_ / patchSize;
}

void RegGraph::RemoveCoarseScales() {
  // delete nodes above the finest scale
  int fine_level_nodes = GetNNodesUpTo(finest_scale_ + scale_step_);
  nodes_.erase(nodes_.begin() + fine_level_nodes, nodes_.end());
  
  // remove parent connections for each fine scale node
  for (std::vector<RegNode>::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
    it->RemoveParentLink();
  }

  // update metadata
  n_nodes_ = static_cast<int>(nodes_.size());
  n_scales_ = 1;
  //n_level_nodes_.erase(n_level_nodes_.begin(), n_level_nodes_.end());
  n_level_nodes_.clear();
  n_level_nodes_[finest_scale_] = n_nodes_;
}