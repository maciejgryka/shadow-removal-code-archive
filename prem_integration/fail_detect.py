import sys

import PIL.Image
sys.modules['Image'] = PIL.Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


PATCH_SIZE = 16
BLUR_SIGMA = 3

def get_grid_image(size):
	im = np.zeros(size)
	for y in range(im.shape[0]):
		if y % PATCH_SIZE == 0 or (y + 1) % PATCH_SIZE == 0:
			im[y, :] = 1.0
	for x in range(im.shape[1]):
		if x % PATCH_SIZE == 0 or (x + 1) % PATCH_SIZE == 0:
			im[:, x] = 1.0
	return im


def color_busy_squares(image):
	copy = np.zeros(image.shape)
	for y in range(0, image.shape[0], PATCH_SIZE):
		for x in range(0, image.shape[1], PATCH_SIZE):
			square = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
			if square.sum() > 10:
				copy[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 1.0
	return copy


def fail_detect(shad, unshad):
	sdy, sdx = np.gradient(shad)
	gs = abs(sdy) + abs(sdx)

	udy, udx = np.gradient(unshad)
	gu = abs(udy) + abs(udx)

	grid = get_grid_image(shad.shape)

	ngs = gu / gaussian_filter(gu, sigma=BLUR_SIGMA)
	ngu = gs / gaussian_filter(gs, sigma=BLUR_SIGMA)
	errors = (ngs - ngu) * grid
	errors = errors * (errors > 0.7)
	return color_busy_squares(errors)


if __name__ == "__main__":
	img_name = sys.argv[1]
	shad = plt.imread(img_name + "_shad.png")[:,:,0]
	unshad = plt.imread(img_name + "_unshad.png")[:,:,0]
	squares = fail_detect(shad, unshad)
	plt.gray()
	plt.imsave(img_name + "_errors.png", squares)
	# produce a negative
	noerrors = 1.0 - squares
	plt.imsave(img_name + "_noerrors.png", squares)
