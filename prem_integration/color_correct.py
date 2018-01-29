# Implementation of
#     Color Transfer between Images
#     by Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley
#     Computer Graphics and Applications, IEEE 21.5 (2001): 34-41.
#     http://www.thegooch.org/Publications/PDFs/ColorTransfer.pdf

import sys

import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from pymorph import erode, dilate, sedisk

# RGB_TO_LMS = np.array([
#     [0.3811, 0.5783, 0.0402],
#     [0.1967, 0.7244, 0.0782],
#     [0.0241, 0.1288, 0.8444]
# ])

# LMS_TO_RGB = np.array([
#     [ 4.4679, -3.5873,  0.1193],
#     [-1.2186,  2.3809, -0.1624],
#     [ 0.0497, -0.2439,  1.2045]
# ])

# # LMS_TO_LAB = np.array([
# #     [1.0/np.sqrt(3.0), 0.0, 0.0],
# #     [0.0, 1.0/np.sqrt(6.0), 0.0],
# #     [0.0, 0.0, 1.0/np.sqrt(2.0)]
# # ]).dot(np.array([
# #     [1.0,  1.0,  1.0],
# #     [1.0,  1.0, -2.0,],
# #     [1.0, -1.0,  0.0]
# # ]))
# LMS_TO_LAB = np.array([
#     [0.57735027,  0.57735027,  0.57735027],
#     [0.40824829,  0.40824829, -0.81649658],
#     [0.70710678, -0.70710678,  0.0       ]
# ])

# # LAB_TO_LMS = np.array([
# #     [1.0,  1.0,  1.0],
# #     [1.0,  1.0, -1.0],
# #     [1.0, -2.0,  0.0]
# # ]).dot(np.array([
# #     [np.sqrt(3.0)/3.0, 0.0, 0.0],
# #     [0.0, np.sqrt(6.0)/6.0, 0.0],
# #     [0.0, 0.0, np.sqrt(2.0)/2.0]
# # ]))
# LAB_TO_LMS = np.array([
#     [ 0.57735027,  0.40824829,  0.70710678],
#     [ 0.57735027,  0.40824829, -0.70710678],
#     [ 0.57735027, -0.81649658,  0.0       ]
# ])


# def rgb_to_lab(rgb):
#     """
#     Take a numpy array representing an RGB image and return the same image in
#     the Lab space.

#     """
#     lab = np.array(rgb)
#     for y in range(lab.shape[0]):
#         for x in range(lab.shape[1]):
#             lab[y, x, :] = LMS_TO_LAB.dot(np.log(RGB_TO_LMS.dot(rgb[y, x, :])))
#     return lab


# def lab_to_rgb(lab):
#     """
#     Take a numpy array representing an Lab image and return the same image in
#     the RGB space.

#     """
#     rgb = np.array(lab)
#     for y in range(rgb.shape[0]):
#         for x in range(rgb.shape[1]):
#             rgb[y, x, :] = LMS_TO_RGB.dot(np.exp(LAB_TO_LMS.dot(lab[y, x, :])))
#     return rgb


def get_mean(image, mask):
    """Return the mean of all the masked elements in the image."""
    return np.mean(image[mask==1.0])


def get_std(image, mask):
    """Return standard deviation of all the masked elements in the image."""
    return np.std(image[mask==1.0])


def color_match_region(image, mask, max_reruns=0):
    """
    Performs color matching to make the masked image region have similar
    color distribution to the rest of the image.

    """
    N_CHANNELS = 3
    assert(image.shape[2] == N_CHANNELS)

    lab = rgb2lab(image)
    mask = np.array(mask)
    mask_neg = 1.0 - mask
    
    # get target and source color distributions
    target_mean = np.array([
        get_mean(lab[:,:,0], mask),
        get_mean(lab[:,:,1], mask),
        get_mean(lab[:,:,2], mask)
    ])
    target_std = np.array([
        get_std(lab[:,:,0], mask),
        get_std(lab[:,:,1], mask),
        get_std(lab[:,:,2], mask)
    ])

    source_mean = np.array([
        get_mean(lab[:,:,0], mask_neg),
        get_mean(lab[:,:,1], mask_neg),
        get_mean(lab[:,:,2], mask_neg)
    ])
    source_std = np.array([
        get_std(lab[:,:,0], mask_neg),
        get_std(lab[:,:,1], mask_neg),
        get_std(lab[:,:,2], mask_neg)
    ])

    # modify the target distribution
    for c in range(N_CHANNELS):
        lab[:,:,c][mask == 1.0] = (
            (lab[:,:,c][mask == 1.0] - target_mean[c]) # subtract target mean
            * target_std[c] / source_std[c] # modify std
            + source_mean[c] # add source mean
        )

    result = lab2rgb(lab)

    if np.any(result > 1.0) and (max_reruns > 0):
        # exclude pixels where any of the channels is saturated
        mask -= np.logical_or(
            np.logical_or(result[:,:,0] > 1.0, result[:,:,1] > 1.0),
            result[:,:,2] > 1.0
        )
        return color_match_region(image, mask, max_reruns-1)
    else:
        return result


if __name__ == '__main__':
    img_path = sys.argv[1]
    mask_path = sys.argv[2]
    max_reruns = 0
    if len(sys.argv) == 4:
        max_reruns = int(sys.argv[3])

    print sys.argv[1]

    img = plt.imread(img_path)
    mask = plt.imread(mask_path)[:,:,0]
    
    matched = color_match_region(img, mask, max_reruns)

    plt.gray()
    plt.imshow(matched)
    plt.show()
