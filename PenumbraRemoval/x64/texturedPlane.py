from mayavi import mlab
from tvtk.api import tvtk
from numpy import genfromtxt
import numpy as np
import random

import pdb

def resize(plane, scale):
	ct = plane.center
	p1 = plane.point1
	p2 = plane.point2
	
	plane.set(point1 = ct + (ct-p1)*scale)
	plane.set(point2 = ct + (ct-p2)*scale)
	
	return plane

def draw_bilboard(fig, texture_name, pos=(0,0,0), scale=1.0):
	plane = tvtk.PlaneSource(center=pos)
	# pdb.set_trace()
	# plane = resize(plane, scale)

	reader = tvtk.PNGReader()
	reader.set_data_scalar_type_to_unsigned_char()
	reader.set(file_name=texture_name)

	plane_texture = tvtk.Texture()
	plane_texture.set_input(reader.get_output())
	plane_texture.set(interpolate=0)
	# pdb.set_trace()

	map = tvtk.TextureMapToPlane()
	map.set_input(plane.get_output())

	plane_mapper = tvtk.PolyDataMapper()
	plane_mapper.set(input=map.get_output())

	p = tvtk.Property(opacity=1.0, color=(1, 0, 0))
	plane_actor = tvtk.Actor(mapper=plane_mapper, texture=plane_texture)
	fig.scene.add_actor(plane_actor)


points = genfromtxt('C:/Work/VS2010/PenumbraRemoval/x64/data/tree_patches/projected_labels.csv', delimiter=',')
count = range(points.shape[0])
step = 50
# pdb.set_trace()
points = points[::step]
# pdb.set_trace()
count = count[::step]
points *= 10

# pdb.set_trace()
	
im_name = 'matte_gt.png'
pos = (5, 1, 0)
v = mlab.figure()

psize = 0.02

for [p, cn] in zip(points, count):
	im_name = 'C:/Work/VS2010/PenumbraRemoval/x64/data/tree_patches/patch%i.png'%(cn)
	draw_bilboard(v, im_name, p, scale=0.1)

# Choose a view angle, and display the figure
mlab.view(0, 0, 40)
mlab.show()