#ifndef REG_GRAPH_H
#define REG_GRAPH_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <MRFEnergy.h>

#include "RegGraphParams.h"
#include "RegNode.h"

class RegNode;
class RegLabel;

class RegGraph {
public:

    RegGraph();
    RegGraph(int nScales, int finestScale, int scaleStep, int finestWidth, int finestHeight);
	
    ~RegGraph() {}

    void ConnectLevel(int level, const std::vector<int>& maskedNodes);
    
    void ConnectLevels(
        int coarseLevel,
        const std::vector<int>& coarseNodes,
        int fineLevel,
        const std::vector<int>& fineNodes
    );

    // return number of nodes at the given level
    int GetNNodesAtLevel(int level);

    // number of nodes at all finer levels (exclusive)
    int GetNNodesUpTo(int level) const;

    // return node from <level> that lies in <x, y> node-coordinates
    RegNode* GetNode(int level, int x, int y);

    // returns <index>-th node (can be ANY scale)
    RegNode* GetNode(int index) { return &(nodes_[index]); }

    // returns <index>-th connected node (can be ANY scale)
    RegNode* GetConnectedNode(int index) { return connected_nodes_[index]; }

    // return node from a given <level> that contains the pixel at <x, y> image coordinates
    RegNode* GetNodeAtPixel(int level, int px, int py);

    // deletes all nodes at scales coarser than finest_scale_
    void RemoveCoarseScales();

    // return image coordinates of the top-left pixel of the given node
    int GetPixelCoordX(const RegNode& node) const;
    int GetPixelCoordY(const RegNode& node) const;
  
    // prepare graph for regularization:
    //   - count connected nodes and put (pointers to) them in a separate array
    //   - set boundary conditions
    void Clean(bool set_bound = true);

    // collapses the graph into one scale (finest_scale_), where nodes include 
    // sliced up labels from the finest as well as all higher scales
    void Collapse(float unary_cost_scaling);

    EigenMat Regularize(EigenMat* out_variances = NULL);

    // returns the matrix of pairwise costs between two given nodes
    static void GetEdgeMatrix(
        TypeGeneral::REAL* edge_term,
        RegNode* n1,
        RegNode* n2,
        EdgeType relationship
    );

    int n_nodes() const { return n_nodes_; }
    int n_scales() const { return n_scales_; }
    int finest_scale() const { return finest_scale_; }
    int scale_step() const { return scale_step_; }

private:
    // sets all unconnected neighbors to constant value and connects them to node
    void SetUnconnectedNeighborsToConstant(RegNode* node, float value);
    // set a specified node to constant label with 0 cost
    void SetToConstant(RegNode* node, float value);

    bool HasUnconnectedNeighbors(const RegNode& node);

    // return the level that this node is at
    int GetNodeLevel(const RegNode& node) const;

    // return id of the north peer (even if there's no connection)
    // if there's no north peer (the node is at the top of the image) return -1
    int GetNorthPeerId(const RegNode& node);
    int GetEastPeerId(const RegNode& node);
    int GetSouthPeerId(const RegNode& node);
    int GetWestPeerId(const RegNode& node);

    // return the row/column of the graph that the node is on
    // the row/column is with respect to the level of the graph that the node is at
    int GetNodeColumn(const RegNode& node) const;
    int GetNodeRow(const RegNode& node) const;

    // get the width of the level that the given node is at (number of nodes horizontally)
    int GetLevelWidth(const RegNode& node) const;
    // get the height of the level that the given node is at (number of nodes vertically)
    int GetLevelHeight(const RegNode& node) const;

    // add a specified type of edge between two nodes
    // in case of parent-child edge (edge_type = EDGE_Z), parent_offset is required
    static void AddEdge(
        RegNode* n1,
        RegNode* n2,
        EdgeType edge_type,
        ParentOffset parent_offset = ParentOffset(-1, -1)
    );
	
    static bool IsPowerOfTwo(int x) { return (x & (x - 1)) == 0; }
    static bool IsDivisibleBy(int number, int divider) {
        return number % divider == 0;
    }

    // hacky function for memory management
    void ReallocIfNeeded(
        TypeGeneral::REAL** edge_term,
        int* desired_alloc_size,
        int* current_alloc_size,
        bool* initialized
    );

    int n_scales_;          // number of scales (levels) in the graph
    
    int finest_scale_;      // finest scale (0: 1x1 patches, 
                            //               1: 2x2 patches,
                            //               2: 4x4 patches etc.)
    
    int scale_step_;        // difference between neighboring scales

	int width_;             // width at scale 0 (pixel dimensions of the image)
	int height_;            // height at scale 0

	int n_nodes_;           // total number of nodes at all scales

	int n_connected_nodes_;	// number of connected nodes

	std::map<int, int> n_level_nodes_;  // number of nodes at each level

    // vector of all nodes in the graph
	std::vector<RegNode> nodes_;
    // root of the whole graph (top-left node at the finest scale) is nodes_[0]
    // then nodes are stored in row-wise order

	std::vector<RegNode*> connected_nodes_;	// vector of (pointers to) all connected nodes
};

#endif // REG_GRAPH_H