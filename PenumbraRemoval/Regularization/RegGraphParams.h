#ifndef REG_GRAPH_PARAMS
#define REG_GRAPH_PARAMS

#include <vector>


#define N_CHILDREN 4
#define N_PEERS 4


typedef std::pair<int, int> ParentOffset;

enum EdgeType {
  EDGE_INVALID = 0,
	EDGE_X,		// horizontal edge (in the left-right order)
	EDGE_Y,		// vertical edge (in the top-bottom order)
  EDGE_Z    // edge across scales (parent-child)
};

#endif REG_GRAPH_PARAMS