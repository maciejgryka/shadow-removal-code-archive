import os
import sys
import re
import errno
import subprocess
import shutil
import colorsys
import random

import numpy as np
from numpy.linalg import norm, lstsq, det
import matplotlib.pyplot as plt

import PIL.Image
sys.modules['Image'] = PIL.Image

from scipy.ndimage import gaussian_filter
from scipy.misc import imresize

from pymorph import erode, dilate, sedisk

from fail_detect import fail_detect

import cppio
import filterbank

from color_solver import color_solver


OPTIONS_COMMON = ["train", "test", "finest_scale", "n_scales", "scale_step",
"feature_intensity", "feature_gradient_orientation",
"feature_gradient_magnitude", "feature_gradient_xy",
"feature_distance_transform", "feature_polar_angle", "feature_gmatte",
"data_file", "labels_file", "data_dir", "pca_file", "ensemble_file",
"uniform_finest", "inpaint_exec", "align_rot", "align_trans", "prem_exec",
"auto_train", "plane_inpaint", "regnn_exec",]

OPTIONS_TRAINING = ["n_dim_out", "n_trees", "tree_depth", "fraction_dim_trials",
"n_thresh_trials", "bag_prob", "min_sample_count", "n_training_samples",
"generate_data", "compute_pca", "image_folder", "image_list_file",]

OPTIONS_TEST = ["unary_cost", "unary_cost_scaling", "image_folder", "results_dir",
"image_list_file", "relationship_weight_peer", "relationship_weight_parent",
"pairwise_weight_alpha", "pairwise_weight_beta", "keep_top_n_labels", "gs_exec",
"gs_postprocess", "deblock_opt_file_source", "deblock_opt_file_target",
"write_guessed_mattes", "mask_exec", "grid_offset",
"grid_offset_x", "grid_offset_y", "patchmatch_exec", "set_bound", "n_iters",
"n_training_images_auto", "overseg_exec",]

LINE_PATTERN = "^(?P<key>[^\s]+)\s(?P<value>.+)[\s]*$"

# patche size used by PatchMatch
PM_PATCH_SIZE = 7


def myimshow(img):
  fig = plt.imshow(img, vmin=0.0, vmax=1.0)
  fig.set_interpolation("nearest")
  plt.gray()


def imsave_rgb(path, img):
  if len(img.shape) == 2 or img.shape[2] == 1:
    img = np.dstack([img, img, img])
  pil_img = PIL.Image.fromarray(np.uint8(img*255))
  pil_img.save(path)


def imread_channel(path, channel=0):
    img = plt.imread(path)
    if len(img.shape) == 3 and img.shape[2] > 1:
        img = img [:,:,channel]
    return img


def create_dir(path):
  """Creates the specified directory and silently fails if it already exists."""
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST:
      pass
    else: raise


def run_exec(exec_path, args):
    """
    Run the executable pointed to by exec_path with given arguments.
    args should be a list of arguments to be passed in.

    """
    if not hasattr(args, "__iter__"):
        print "ERROR: args has to be iterable"
        sys.exit()
    ts = subprocess.Popen("%s %s" % (exec_path, " ".join(args)))
    print "%s %s" % (exec_path, " ".join(args))
    retcode = ts.wait()


def write_iterable_to_file(iterable, filename):
    """Writes a text file with each line being an element in the iterable."""
    f = open(filename, 'w');
    for item in iterable:
        f.write('%s\n' % str(item))
    f.close()


def mask_gt(test_dir, im_name, channel=0):
    file_name = os.path.join(test_dir, ''.join([im_name, '_smask.png']))
    im = plt.imread(file_name)
    return im[:,:,channel]


def matte_gt(test_dir, im_name, channel=0):
    file_name = os.path.join(test_dir, ''.join([im_name, '_matte_gt.png']))
    im = plt.imread(file_name)
    return im[:,:,channel]


def noshad_gt(test_dir, im_name, channel=0):
    file_name = os.path.join(test_dir, ''.join([im_name, '_noshad.png']))
    im = plt.imread(file_name)
    return im[:,:,channel]


def matte_result(output_dir, k, im_name):
    size_dir = 'size%i' % k
    im_name = ''.join([im_name, '_matte.png'])
    file_name = os.path.join(output_dir, size_dir, 'results', im_name)
    return plt.imread(file_name)


def unshad_result(results_folder_name, im_name):
    im_name = ''.join([im_name, '_unshad.png'])
    file_name = os.path.join(results_folder_name, im_name)
    return plt.imread(file_name)


def msd(arr1, arr2):
    return ssd(arr1, arr2)/len(arr1)


def gradient_orientation_image(im):
    dy, dx = np.gradient(im)
    return np.arctan2(dy, dx)


def gradient_xy(im):
    dy, dx = np.gradient(im)
    return dx, dy


def ssd(arr1, arr2):
    sq_diff = (arr1 - arr2) * (arr1 - arr2)
    return sq_diff.sum()


def write_options(options_file, options_dict, base_options=''):
    op_file = open(options_file, 'w')
    if len(base_options) != 0:
      bop_file = open(base_options, 'r')
      bop_lines = bop_file.readlines()
      bop_file.close()
      op_file.writelines(bop_lines)
    for k, v in options_dict.iteritems():
      op_file.write('%s %s\n' % (k, v))
    op_file.close()


def read_options(options_filename, extra_options=None):
    """
    Read an options file and return a dictionary. Each option can use a previously-
    defined option's value using python's standard %(name)s syntax.

    NOTE: Order matters!

    """
    op_file = open(options_filename, 'r')
    op_lines = op_file.readlines()
    op_file.close()
    options = {}
    if extra_options is not None:
        options.update(extra_options)
    # discard comments
    for line in op_lines:
        if line.startswith("//"):
            continue
        # if the line is valid, save the option
        rem = re.match(LINE_PATTERN, line)
        if rem is not None:
            key = rem.group("key")
            value = rem.group("value")
            # a special case for local_options_file key to read the
            # specified options file first
            if key == "local_options_file":
                local_options = read_options(value, options)
                # now copy all the keys from local_options to options allowing
                # for substitution of already-defined values
                for key_local in local_options.keys():
                    options[key_local] = local_options[key_local] % options
            else:
                options[key] = value % options
    return options


def copy_options(option_list, source, target):
    for key in source.keys():
        if key in option_list:
            target[key] = source[key]
    return target


def get_file_names(directory, condition=lambda f: True):
    """Return a list of files/folders in the given directory.

    Optional condition is a boolean function that determines whether to include
    the file.

    """
    return [f for f in os.listdir(directory) if condition(f)]


def separate_image_names(file_names, pattern="^(?P<img_name>.*)_shad.png$"):
    """
    Return a list of image names from the given list of file names.
    If a custom regex is given it must contain a named group "img_name".

    """
    image_names = []
    for file_name in file_names:
        rem = re.match(pattern, file_name)
        if rem is not None:
            image_names.append(rem.group("img_name"))
    return image_names
    # return [re.match(pattern, file_name).group("img_name") for file_name in file_names]


def read_lines_into_list(file_path):
    f = open(file_path, "r")
    return [line.rstrip() for line in f.readlines()]


def create_mattes(dir_name):
    """
    Create mattes for images in the given folder where both _shad and _noshad
    image exists and matte is not present.

    """
    shads = [
        img_name.split("_shad")[0]
        for img_name in  os.listdir(dir_name)
        if img_name.endswith("_shad.png")
    ]
    noshads = [
        img_name.split("_noshad")[0]
        for img_name in  os.listdir(dir_name)
        if img_name.endswith("_noshad.png")
    ]
    mattes = [
        img_name.split("_matte")[0]
        for img_name in  os.listdir(dir_name)
        if img_name.endswith("_matte.png")
    ]
    for img_name in shads:
        print img_name
        if img_name in noshads:
            shad = imread_channel(os.path.join(dir_name, img_name + "_shad.png"), 0)
            noshad = imread_channel(os.path.join(dir_name, img_name + "_noshad.png"), 0)
            matte = shad / noshad
            imsave_rgb(os.path.join(dir_name, img_name + "_matte.png"), matte)


def clean_mask(mask):
    # first erode to get rid of noise
    mask = erode(mask, sedisk(2))
    # then dilate more to capture a slightly larger area
    mask = dilate(mask, sedisk(16))

    return mask


def take_single_channel(image, channel=0):
    if len(image.shape) == 3:
        return image[:,:,channel]
    else:
        return image


def make_mask(shad, noshad):
    # if the images are rgb, only take the first channel
    shad = take_single_channel(shad)
    noshad = take_single_channel(noshad)
    # element-wise division
    return clean_mask((shad/noshad) < 1)


def make_mask_for_name(img_name, dir_path="."):
    shad = plt.imread(os.path.join(dir_path, '{0}_shad.png'.format(img_name)))
    shad = take_single_channel(shad)
    noshad = plt.imread(os.path.join(dir_path, '{0}_noshad.png'.format(img_name)))
    noshad = take_single_channel(noshad)

    mask = clean_mask((shad/noshad) < 1)
    plt.imsave(os.path.join(dir_path, '{0}_smask.png'.format(img_name)), mask)


def make_maskp_for_name(img_name, dir_path="."):
    shad = plt.imread(os.path.join(dir_path, '{0}_shad.png'.format(img_name)))
    shad = take_single_channel(shad)
    noshad = plt.imread(os.path.join(dir_path, '{0}_noshad.png'.format(img_name)))
    noshad = take_single_channel(noshad)

    matte = shad/noshad
    dx, dy = np.gradient(matte)
    maskp = gaussian_filter(abs(dx) + abs(dy), sigma=20) > 0.014
    # if the overall count of white pixels in maskp is smaller than 100,
    # just use smask
    if np.sum(maskp) < 500:
        maskp = plt.imread(os.path.join(dir_path, '{0}_smask.png'.format(img_name)))
        if len(maskp.shape) == 3:
            maskp = maskp[:,:,0]
    plt.imsave(os.path.join(dir_path, '{0}_maskp.png'.format(img_name)), maskp)


def create_missing_masks(dir_path):
    """
    Automatically create masks for all shad-noshad image pairs in the
    given location.

    """
    shads = [
        shad.split('_shad')[0]
        for shad in os.listdir(dir_path)
        if shad.endswith('_shad.png')
    ]

    noshads = [
        noshad.split('_noshad')[0]
        for noshad in os.listdir(dir_path)
        if noshad.endswith('_noshad.png')
    ]

    masks = [
        mask.split('_smask')[0]
        for mask in os.listdir(dir_path)
        if mask.endswith('_smask.png')
    ]

    maskps = [
        mask.split('_maskp')[0]
        for mask in os.listdir(dir_path)
        if mask.endswith('_maskp.png')
    ]

    plt.gray()

    for img in shads:
        if img not in masks:
            make_mask_for_name(img, dir_path)
        if img not in maskps:
            make_maskp_for_name(img, dir_path)


def is_shad(name):
    """Return True if the given name is a shadow image."""
    return name.endswith("_shad.png")


def is_matte(name):
    """Return True if the given name is a matte image."""
    return name.endswith("_matte.png")


class WrongDepthException(Exception):
    pass


def make_rgb(image):
    """Returns an 3D (RGB) matrix from single-channel one."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        out = np.zeros([image.shape[0], image.shape[1], 3])
        out[:,:,0] = image
        out[:,:,1] = image
        out[:,:,2] = image
        return out
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image


def mean_rgb(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image.mean()

    return np.array([
        image[:,:,0].mean(),
        image[:,:,1].mean(),
        image[:,:,2].mean(),
    ])


def replace_with_color(image, mask, color=[255, 255, 255]):
    """
    Takes an image and a mask (binary, same size as image) and an optional
    color and returns an image where masked region is replaced with color.

    """
    for c in range(image.shape[2]):
        temp = image[:,:,c]
        temp[mask>0] = color[c]
        image[:,:,c] = temp
    return image


def inpaint(img_folder, img_name):
    """
    Use Photoshop to inpaint the file in the given directory. It is assumed that
    the directory contains both <img_name>_shad.png and <img_name>_smask.png
    files. The inpainting fills in the area of <img_name>_shad.png masked by
    <img_name>_smask.png and saves the result as <img_name>_gunshadp.png

    Also assumes that .jsx script for inpainting is in the current directory.

    """
    PS_SCRIPT_DIR = "C:\\Program Files\\Adobe\\Adobe Photoshop CS6 (64 Bit)\\Presets\\"
    # first read the .jsx file
    script_contents = open("inpaint_file_template.jsx", "r").read()
    # now fill in the directory and the filename (make sure the directory uses
    # double slashes)
    img_folder = img_folder.replace("\\", "\\\\")
    if not img_folder.endswith("\\\\"):
        img_folder += "\\\\"
    script_contents = script_contents.replace("***img_folder***", img_folder)
    script_contents = script_contents.replace("***img_name***", img_name)
    script_out = open(os.path.join(PS_SCRIPT_DIR, "inpaint_file.jsx"), "w")
    script_out.write(script_contents)
    script_out.close()

    run_exec("inpaint_file.exe", [os.path.join(os.getcwd(), "dummy.jpg")])


def pm_apply(img, mask, list_path):
    points = np.loadtxt(list_path, dtype="int", delimiter=" ")
    out = np.array(img)
    for pt in points:
        out[pt[1], pt[0]] = img[pt[3], pt[2]]
    return out


def rgb_to_gray(img):
    return 0.33  * img[:,:,0] + 0.34 * img[:,:,1] + 0.33 * img[:,:,2]
    # return 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]


def remove_if_exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def inpaint2(img_folder, img_name, options):
    img_folder = options["image_folder"]
    shad_path = os.path.join(img_folder, img_name + "_shad.png")
    mask_path = os.path.join(img_folder, img_name + "_smask.png")
    if "pre_training" in options and int(options["pre_training"]) and int(options["auto_train"]):
        mask_path = os.path.join(img_folder, img_name + "_inpaint_mask.png")


    shad = plt.imread(shad_path)
    shad_r = shad[:,:,0]
    mask = plt.imread(mask_path)[:,:,0]
    # create a normalized-intensity image
    nimg1 = shad_r / gaussian_filter(shad_r, sigma=1)
    nimg1 /= np.max(nimg1)
    # blur the normalized-intensity image for robustness
    nimg1 = gaussian_filter(nimg1, sigma=1)

    cs = np.sum(shad, axis=2)
    nimg2 =  shad / np.dstack([cs, cs, cs])

    nimg3 = nimg2[:,:,2] / nimg2[:,:,0]

    nimg2 = nimg2[:,:,1] / nimg2[:,:,0]

    nimg2[np.isnan(nimg2)] = 0.0
    nimg2[np.isinf(nimg2)] = 1.0
    nimg2 -= np.min(nimg2)
    nimg2 /= np.max(nimg2)

    nimg3[np.isnan(nimg3)] = 0.0
    nimg3[np.isinf(nimg3)] = 1.0
    nimg3 -= np.min(nimg3)
    nimg3 /= np.max(nimg3)

    # nimg = np.dstack([nimg1, nimg2, nimg3])

    import filterbank
    sigmas = [4]
    response = filterbank.get_responses(shad[:,:,0], sigmas=sigmas)
    nimg = np.dstack([nimg1, nimg2] + response)

    nimg_path = os.path.join(img_folder, img_name + "_nimg.png")

    # save nimg for use by patchmatch
    cppio.write(mask_path + ".bin", mask, 'uint8')
    cppio.write(shad_path + ".bin", shad, 'float32')
    cppio.write(nimg_path + ".bin", nimg, 'float32')

    # prepare and run patchmatch
    results_dir = options["results_dir"]
    guess_path = os.path.join(img_folder, img_name + "_gunshadp.png")
    run_exec(
        options["regnn_exec"],
        [
            shad_path + ".bin",
            nimg_path + ".bin",
            mask_path + ".bin",
            guess_path,
        ]
    )

    # cleanup
    os.remove(shad_path + ".bin");
    os.remove(mask_path + ".bin");
    os.remove(nimg_path + ".bin");
    # ann_list_path = os.path.join(results_dir, img_name + "_ann_list.txt")
    # run_exec(
    #     options["patchmatch_exec"],
    #     [
    #         nimg_path,
    #         mask_path,
    #         os.path.join(results_dir, img_name + "_ann.bmp"),
    #         ann_list_path,
    #     ]
    # )
    # # cleanup after patchmatch
    # remove_if_exists(os.path.join(results_dir, img_name + "_ann.raw"))
    # remove_if_exists(os.path.join(results_dir, img_name + "_ann.bmp"))
    # remove_if_exists(os.path.join(img_folder, img_name + "_nimg.txt"))
    # remove_if_exists(os.path.join(img_folder, img_name + "_nimg.raw"))
    # # apply the patchmatch result
    # guess = pm_apply(shad, mask, ann_list_path)
    # imsave_rgb(guess_path, guess)


def get_patch(image, coords):
    return image[coords[1]:coords[1]+PM_PATCH_SIZE, coords[0]:coords[0]+PM_PATCH_SIZE]


def pre_train_tasks(options):
    create_missing_masks(options["image_folder"])
    image_dir = options["image_folder"]
    img_names = read_lines_into_list(options["image_list_file"])
    # create the first-guess images and the corresponding mattes
    for img_name in img_names:
        gunshad_path = os.path.join(image_dir, img_name + "_gunshadp.png")
        gmatte_path = os.path.join(image_dir, img_name + "_gmatte.png")
        if int(options["feature_gmatte"]) or not (os.path.isfile(gmatte_path) or os.path.isfile(gunshad_path)):
            shad_path = os.path.join(image_dir, img_name + "_shad.png")
            mask_path = os.path.join(image_dir, img_name + "_smask.png")
            print "inpainting..."
            options["pre_training"] = 1
            # inpaint2(options["image_folder"], img_name, options)
            inpaint(options["image_folder"], img_name)
            print "done"
            shad = imread_channel(shad_path)
            mask = imread_channel(mask_path)
            gunshad = imread_channel(gunshad_path)
            import pdb; pdb.set_trace()
            # blur guessed unshad and produce the guessed matte
            gunshad = gaussian_filter(gunshad, sigma=5)
            # make sure that at each pixel the unshadowed is at least as bright
            # as the shadowed image
            gunshad = np.maximum(gunshad, shad)
            imsave_rgb(os.path.join(image_dir, img_name + "_gunshad_blur.png"), gunshad)
            shad = imread_channel(shad_path)
            gmatte = shad / gunshad
            # make sure we don't modify the out-of-shadow pixels
            gmatte[mask==0] = 1.0
            imsave_rgb(os.path.join(image_dir, img_name + "_gmatte.png"), gmatte)


def create_shad(matte, target):
    """
    Creates a shadowed image given target (to be shadowed) and matte.
    The matte can be smaller that the target in which case it will be used at
    some random location.

    """
    # first get a bounding box of the shadow matte
    mask = np.array(matte < 1, dtype=int)
    mask = dilate(erode(mask, sedisk(3)), sedisk(3))
    left, upper, right, lower = PIL.Image.fromarray(mask).getbbox()

    # now cut it out
    mh, mw = matte.shape[:2]
    matte_bbox = matte[upper:lower, left:right]

    # import pdb; pdb.set_trace()
    # get new dimensions
    mh, mw = matte_bbox.shape[:2]
    th, tw = target.shape[:2]

    new_matte = np.ones(target.shape)

    # get random position to insert the matte
    matte_x = matte_y = 0
    if mh < th:
        matte_y = (th - mh) * np.random.random()
    if mw < tw:
        matte_x = (tw - mw) * np.random.random()

    new_matte[matte_y:matte_y+mh, matte_x:matte_x+mw] = matte_bbox
    return new_matte * target


def remove_if_exist(file_names):
    [os.remove(fn) for fn in list(file_names) if os.path.isfile(fn)]


def bootstrap_training(img_name, options):
    """
    Create a training set from the test image and train on it.

    """
    training_dir = options["training_image_folder"]
    results_dir = options["results_dir"]

    training_images = get_file_names(training_dir, is_shad)
    training_names = separate_image_names(training_images)

    n_training_images = int(options["n_training_images_auto"])
    sub_training_names = random.sample(training_names, n_training_images)

    # create a dir where we will store auto training images
    auto_train_dir = os.path.join(results_dir, img_name)
    create_dir(auto_train_dir)

    shad_path = os.path.join(options["image_folder"], img_name + "_shad.png")
    mask_path = os.path.join(options["image_folder"], img_name + "_smask.png")

    shad = imread_channel(shad_path)
    mask = imread_channel(mask_path)

    plt.gray()

    removed_names = []

    for tr_name in sub_training_names:
        tr_shad_path = os.path.join(training_dir, tr_name + "_shad.png")
        noshad_path = os.path.join(training_dir, tr_name + "_noshad.png")
        matte_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_matte.png")

        atr_shad_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_shad.png")
        atr_noshad_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_noshad.png")
        atr_mask_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_smask.png")
        atr_inpaint_mask_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_inpaint_mask.png")

        if os.path.isfile(atr_shad_path) and os.path.isfile(atr_noshad_path) and os.path.isfile(atr_mask_path):
            continue

        matte = imread_channel(tr_shad_path) / imread_channel(noshad_path)
        matte[matte > 1.0] = 1.0
        atr_shad = create_shad(matte, shad)

        atr_mask = make_mask(atr_shad, shad)
        atr_mask[mask == 1.0] = 0.0
        atr_inpaint_mask = np.array(atr_mask)
        atr_inpaint_mask[mask == 1.0] = 1.0

        # if there's nothing masked, skip
        if np.sum(np.array(atr_mask, dtype=int)) < 100:
            print "removing ", tr_name
            # sub_training_names.remove(tr_name)
            removed_names.append(tr_name)
            continue

        # plt.imsave(matte_path, matte, vmin=0.0, vmax=1.0)
        imsave_rgb(atr_noshad_path, shad)
        imsave_rgb(atr_shad_path, atr_shad)
        imsave_rgb(atr_mask_path, atr_mask)
        imsave_rgb(atr_inpaint_mask_path, atr_inpaint_mask)


    image_list_file_training = os.path.join(auto_train_dir, "image_list.txt")

    atr_options = dict(options)
    atr_options["train"] = 1

    atr_options["image_folder"] = auto_train_dir
    atr_options['image_list_file'] = image_list_file_training
    atr_options["train"] = 1
    atr_options["test"] = 0

    atr_options["generate_data"] = 1
    atr_options["n_training_samples"] = 100
    atr_options["compute_pca"] = 1
    atr_options["n_dim_out"] = 4
    atr_options["n_trees"] =  25
    atr_options["tree_depth"] =  50
    atr_options["fraction_dim_trials"] =  0.1
    atr_options["n_thresh_trials"] =  5
    atr_options["bag_prob"] =  0.5
    atr_options["min_sample_count"] =  8

    sub_training_names = [img_name + "_" + stn for stn in sub_training_names if stn not in removed_names]

    # write training options to file
    training_options_file = os.path.join(auto_train_dir, 'training_options.txt')
    write_options(options_file=training_options_file, options_dict=atr_options)
    write_iterable_to_file(sub_training_names, image_list_file_training)

    # train on this subset
    print 'training with %s...\n' % training_options_file
    pre_train_tasks(atr_options)

    # modify maskps to not include original shadows
    for tr_name in sub_training_names:
        atr_maskp_path = os.path.join(auto_train_dir, tr_name + "_maskp.png")
        atr_maskp = imread_channel(atr_maskp_path)
        atr_maskp[mask == 1.0] = 0.0
        # if there are not enough pixels in maskp, get rid of this image
        if np.sum(np.array(atr_maskp, dtype=int)) < 100:
            removed_names.append(tr_name)
            atr_shad_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_shad.png")
            atr_noshad_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_noshad.png")
            atr_mask_path = os.path.join(auto_train_dir, img_name + "_" + tr_name + "_smask.png")
            remove_if_exist([atr_shad_path, atr_noshad_path, atr_mask_path])
            continue

        imsave_rgb(atr_maskp_path, atr_maskp)

    sub_training_names = [stn for stn in sub_training_names if stn not in removed_names]
    write_iterable_to_file(sub_training_names, image_list_file_training)

    run_exec(options["prem_exec"], [training_options_file])


def pre_test_tasks(options):
    """
    Do the things that need to be done before the main algorithm runs, e.g.
    get the user to paint masks for images that are to be unshadowed, supply
    missing parameters etc.

    """
    image_dir = options["image_folder"]
    results_dir = options["results_dir"]
    img_names = read_lines_into_list(options["image_list_file_original"])

    iter_index = int(options["iter_index"])
    if iter_index != 0:
        iter_dir = os.path.join(results_dir, "iter")
        create_dir(iter_dir)
        img_names_new = [img_name + "_" + str(iter_index) for img_name in img_names]
        img_list_file = os.path.join(iter_dir, "image_list.txt")
        write_iterable_to_file(img_names_new, img_list_file)
        options["image_list_file"] = img_list_file

    for img_name in img_names:
        # get the shadow image
        image_dir_original = options["image_folder_original"]
        shad_path = os.path.join(image_dir_original, img_name + "_shad.png")
        mask_path = os.path.join(image_dir_original, img_name + "_smask.png")
        pmask_path = os.path.join(image_dir_original, img_name + "_pmask.png")

        img_name_new = img_name + "_" + str(iter_index)

        # if we're at a later iteration
        if iter_index != 0:
            # create a direcotry where we're going to store intermediate results
            image_dir = iter_dir
            options["image_folder"] = iter_dir

            # copy the previous iteration's images before overwriting
            shutil.copyfile(
                mask_path,
                os.path.join(iter_dir, img_name_new + "_smask.png")
            )
            if iter_index == 1:
                unshad_path = os.path.join(results_dir, img_name + "_unshad.png")
            else:
                unshad_path = os.path.join(results_dir, img_name + "_" + str(iter_index-1) + "_unshad.png")

            shutil.copyfile(
                unshad_path,
                os.path.join(iter_dir, img_name_new + "_shad.png")
            )

            shad_path = os.path.join(iter_dir, img_name_new + "_shad.png")
            mask_path = os.path.join(iter_dir, img_name_new + "_smask.png")
            pmask_path = os.path.join(iter_dir, img_name_new + "_pmask.png")

            img_name = img_name_new

            if iter_index > 1:
                options["gs_postprocess"] = 0
        # if the mask for this image doesn't exist, ask the user to create it
        if not os.path.isfile(mask_path):
            shad = imread_channel(shad_path)
            # launch Scribbler with this image for the user to draw a shadow mask
            print "drawing mask for " + img_name
            run_exec(options["mask_exec"], [shad_path])

        # produce mask of the unshadowed area if it doesn't already exist
        if not os.path.isfile(pmask_path):
            mask = imread_channel(mask_path)
            pmask = 1.0 - mask
            imsave_rgb(pmask_path, pmask)

        if int(options["auto_train"]) == 1:
            bootstrap_training(img_name, options)

        # if we don't have the first guess image, create it (inpainting)
        gunshad_path = os.path.join(image_dir, img_name + "_gunshadp.png")
        gmatte_path = os.path.join(image_dir, img_name + "_gmatte.png")

        compute_gmatte = int(options["feature_gmatte"]) and not int(options["plane_inpaint"])
        gmatte_exists = bool(os.path.isfile(gmatte_path) and os.path.isfile(gunshad_path))
        if False and compute_gmatte and not gmatte_exists:
            # print "inpainting..."
            # # inpaint2(options["image_folder"], img_name, options)
            # inpaint(options["image_folder"], img_name)
            # print "done"
            # # blur guessed unshad and produce the guessed matte
            # shad = imread_channel(shad_path)
            mask = imread_channel(mask_path)
            # gunshad = imread_channel(gunshad_path)
            # gunshad = gaussian_filter(gunshad, sigma=5)
            # # make sure we don't modify the out-of-shadow pixels
            # gunshad[mask==0] = shad[mask==0]
            # # make sure that at each pixel the unshadowed is at least as bright
            # # as the shadowed image
            # gunshad = np.maximum(gunshad, shad)
            # imsave_rgb(os.path.join(image_dir, img_name + "_gunshad_blur.png"), gunshad)
            gunshad = imread_channel(os.path.join(image_dir, img_name + "_gunshad_blur.png"))
            shad = imread_channel(shad_path)
            gmatte = shad / gunshad
            gmatte[mask==0] = 1.0
            gmatte[gmatte>1.0] = 1.0
            gmatte[~np.isfinite(gmatte)] = 1.0
            imsave_rgb(os.path.join(image_dir, img_name + "_gmatte.png"), gmatte)
    out_dir = results_dir
    test_options_file = os.path.join(out_dir, 'test_options.txt')
    write_options(options_file=test_options_file, options_dict=options)


def post_test_tasks(options):
    """
    Do things that need to be done after the main algorithm runs: remove
    intermediate results and apply the matte to the input image.

    """
    results_dir = options["results_dir"]
    image_dir = options["image_folder"]

    # remove intermediate matte-smoothing images
    [os.remove(os.path.join(results_dir, f))
        for f in os.listdir(results_dir)
        if f.endswith(".png.tif") or f.endswith(".png.jpg")]

    # apply the produced mattes to shadowed images
    img_names = separate_image_names(
        get_file_names(results_dir, is_matte),
        pattern="^(?P<img_name>.+)_matte.png$")

    plt.gray() # make sure all images will be written written in grayscale
    gs_result_path = "result-%s_matte.png.tif"
    for img_name in img_names:
        # if this is a later iteration only process the latest images
        if int(options["iter_index"]) > 0 and not img_name.endswith("_" + str(options["iter_index"])):
            continue
        shad_path = os.path.join(image_dir, img_name + "_shad.png")
        unshad_path = os.path.join(results_dir, img_name + "_unshad.png")
        matte_path = os.path.join(results_dir, img_name + "_matte.png")
        # apply GradientShop deblocking to the produced matte
        if options["gs_postprocess"] == "1":
            shad = plt.imread(shad_path)
            matte = imread_channel(matte_path)
            unshad = shad[:,:,0] / matte
            # find problematic areas
            problems = fail_detect(shad[:,:,0], unshad)
            gsmask_path = os.path.join(results_dir, img_name + "_gsmask.jpg")
            imsave_rgb(gsmask_path, problems)

            gs_opt_source = open(options["deblock_opt_file_source"], "r")
            gs_opt = gs_opt_source.read()
            gs_opt = gs_opt.replace("***gsmask***", gsmask_path)
            gs_opt_out = open(options["deblock_opt_file_target"], "w")
            gs_opt_out.write(gs_opt)
            gs_opt_out.close()

            imsave_rgb(matte_path + ".jpg", matte)
            run_exec(
                options["gs_exec"],
                [
                    options["deblock_opt_file_target"],
                    "imgFN=%s" % (matte_path + ".jpg"),
                ]
            )
            shutil.copyfile(gs_result_path % img_name, matte_path + ".tif")
            os.remove(gs_result_path % img_name)
            os.remove(matte_path + ".jpg")
            matte = PIL.Image.open(matte_path + ".tif")
            matte = matte.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            matte.save(matte_path)
            os.remove(matte_path + ".tif")
        # read the matte and shadow images and produce the final grayscale result
        shad_path = os.path.join(options["image_folder"], img_name + "_shad.png")
        matte = imread_channel(matte_path)
        shad = imread_channel(shad_path)

        if int(options["grid_offset"]):
            new_x = int(options["grid_offset_x"])
            new_y = int(options["grid_offset_y"])
            new_width = shad.shape[1] - 16
            new_height = shad.shape[0] - 16
            shad = shad[new_y:new_y+new_height, new_x:new_x+new_width]

        # do postprocessing to get color result
        # read the mask
        mask_path = os.path.join(options["image_folder"], img_name + "_smask.png")
        mask = imread_channel(mask_path)

        # create the red-channel unshadowed image
        shad = plt.imread(shad_path)
        unshad_r = shad[:, :, 0] / matte
        unshad_path = os.path.join(results_dir, img_name + "_unshad.png")
        imsave_rgb(unshad_path, unshad_r)

        # get green and blue scaling
        small_size = 0.1
        s_g, s_b = color_solver.get_best_channel_scaling(
            np.array(imresize(np.array(shad*255.0, dtype=np.uint8),  size=small_size, interp='cubic') / 255.0, dtype=np.double),
            np.array(imresize(np.array(matte*255.0, dtype=np.uint8), size=small_size, interp='cubic') / 255.0, dtype=np.double)
        )

        matte_g = (matte - 1.0) / s_g + 1.0
        matte_b = (matte - 1.0) / s_b + 1.0

        matte_rgb = np.dstack([matte, matte_g, matte_b])

        matte_rgb[matte_rgb > 1.0] = 1.0
        matte_rgb[np.isnan(matte_rgb)] = 1.0

        unshad = shad / matte_rgb
        unshad[unshad > 1.0] = 1.0
        unshad[np.isnan(unshad)] = 1.0

        matte_rgb_path = os.path.join(results_dir, img_name + "_matte_rgb.png")
        imsave_rgb(matte_rgb_path, matte_rgb)
        imsave_rgb(unshad_path, unshad)

        # cleanup
        files_to_delete = [
            "ndcgs2.csv",
            "ndcgs3.csv",
            "ndcgs4.csv",
            "ndcgs5.csv",
            "null.txt",
        ]
        for f in files_to_delete:
            if os.path.exists(f):
                os.remove(f)
