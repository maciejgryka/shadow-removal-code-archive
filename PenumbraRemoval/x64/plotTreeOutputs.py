from numpy import *
from mayavi.mlab import *

csvfile = 'C:/Work/VS2010/PenumbraRemoval/x64/data/tree_patches/labelsPCA.csv'
lpca = genfromtxt(csvfile, delimiter=',')

mean_label = mean(lpca, axis=0)

# plot
point_size = 0.01
points3d(lpca[:,0], lpca[:,1], lpca[:,2], color=(1,1,1), scale_factor=point_size)
points3d(mean_label[0], mean_label[1], mean_label[2], color=(1,0,0), scale_factor=point_size)