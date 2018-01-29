import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import experiment_common as xc


IMG_DIR = "C:\\Work\\research\\shadow_removal\\experiments\\test_images_real\\"
IMG_NAME = "real162"


GD_DIR = "C:\\Work\\research\\shadow_removal\\prem_integration\\grad_test\\"
GD_PATH = os.path.join(GD_DIR, "grad_dir.png")
GD_MASKED_PATH = os.path.join(GD_DIR, "grad_dir_masked.png")
GD_ANN_PATH = os.path.join(GD_DIR, "grad_dir_ann.bmp")
GD_ANN_LIST_PATH = os.path.join(GD_DIR, "grad_dir_ann.txt")

PM_EXEC = "C:\\Work\\research\\shadow_removal\\experiments\\prem\\PM_Minimal.exe"


def pm_apply(img, mask, l):
    points = np.loadtxt(l, dtype="int", delimiter=" ")
    out = np.array(img)
    for pt in points:
        out[pt[1], pt[0]] = img[pt[3], pt[2]]
    return out


def inpaint(img, mask):
    # normalize image
    grad_dir = img / gaussian_filter(img, sigma=1)
    grad_dir /= np.max(grad_dir)
    # blur it for robustness
    grad_dir = gaussian_filter(grad_dir, sigma=2)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        IMG_NAME = sys.argv[1]

    IMG = IMG_NAME + "_shad.png"
    MASK = IMG_NAME + "_smask.png"

    img = plt.imread(os.path.join(IMG_DIR, IMG))[:,:,0]
    mask = plt.imread(os.path.join(IMG_DIR, MASK))[:,:,0]

    dy, dx = np.gradient(img)
    grad_dir = img / gaussian_filter(img, sigma=1)
    grad_dir /= np.max(grad_dir)
    plt.imshow(grad_dir)
    grad_dir /= np.max(grad_dir)
    grad_dir = gaussian_filter(grad_dir, sigma=1)
    # grad_dir = np.arctan(dy/dx)
    # grad_dir[np.isnan(grad_dir)] = 0.0
    # grad_dir -= np.min(grad_dir)
    # grad_dir /= np.max(grad_dir)
    # grad_dir_masked = np.array(grad_dir)
    # grad_dir_masked[mask > 0.0] = 0.0

    plt.imsave(GD_PATH, grad_dir)
    # plt.imsave(GD_MASKED_PATH, grad_dir_masked)

    xc.run_exec(PM_EXEC, [GD_PATH, os.path.join(IMG_DIR, MASK), GD_ANN_PATH, GD_ANN_LIST_PATH])

    matte = pm_apply(img, mask, GD_ANN_LIST_PATH)
    plt.figure()
    plt.imshow(matte, vmin=0.0, vmax=1.0)
    plt.show()

    # plt.close("all")
    # plt.imshow(grad_dir)
    # plt.figure()
    # plt.imshow(grad_dir_masked)
    # plt.show()
