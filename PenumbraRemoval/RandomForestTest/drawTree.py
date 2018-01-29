from Tkinter import *
from numpy import ones, zeros

import pdb

def read_node(f):
	node_id = int(f.readline())
	is_leaf = int(f.readline())
	split_dim = int(f.readline())
	split_thresh = float(f.readline())
	mean = float(f.readline())
	covar = float(f.readline())
	return node_id

def read_tree(f):
	f.readline()
	tree_id = int(f.readline())
	depth = int(f.readline())
	n_nodes = int(f.readline())
	n_leaves = int(f.readline())
	
	nodes = zeros(2*pow(2,depth)-1)
	# pdb.set_trace()
	for n in range(n_nodes):
		nodes[read_node(f)] = 1
		
	return nodes

def read_forest(file, t=0):
	f = open(file, 'r')
	
	n_trees = f.readline()
	n_dim_in = f.readline()
	n_dim_out = f.readline()
	
	nodes = []
	for tr in range(t+1):
		nodes = read_tree(f)
	
	f.close()
	return nodes

def draw_node(w, x, y, r = 10):
	if (r > 20):
		r = 20;
	w.create_oval(x-r, y-r, x+r, y+r)
	
def draw_binary_tree(depth, nodes):
	tree_w = 128*depth	# tree width
	level_h = 100	# level height
	
	# if (nodes[0] == 0):
		# nodes = ones(2*pow(2,depth)-1)
	
	master = Tk()
	w = Canvas(master, width=tree_w, height=tree_w + level_h/2)
	w.pack()

	for d in range(depth+1):
		row_y = (level_h/2) + d*level_h	# y-coord of the row
		node_dist = tree_w / pow(2,d)	# node offset at this level
		n_nodes = pow(2,d)
		
		for n in range(n_nodes):
			node_id = pow(2,d) - 1 + n
			if (nodes[node_id] > 0):
				node_x = (node_dist/2) + n*node_dist
				draw_node(w, node_x, row_y, node_dist/3)
	mainloop()

draw_binary_tree(4, read_forest('data\\trees.data', 2))

