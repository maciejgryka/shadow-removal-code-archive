# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os
import re
import shutil

import experiment_common as xc

DIR = ("C:\\Work\\research\\shadow_removal\\experiments\\output"
        "\\size7817\\results")
FRAMES_DIR = "frames"
DIR_PATT = "[0-9]{1,2}_[0-9]{1,2}"

IMG_NAMES = [
    # "real109",
    # "real110"
    # "real37"
    # "real38"
    "real42"
]

SUFFIXES = [
    # "_gmatte.png",
    "_matte.png",
    # "_unshad.png"
]

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Need option file as an argument.")
    #     sys.exit()

    # get list of all directories that match the pattern
    dirs = [
        f for f in os.listdir(DIR) if
        os.path.isdir(os.path.join(DIR, f))
        and
        re.search(DIR_PATT, f) is not None
    ]

    for img_name in IMG_NAMES:
        target_dir = os.path.join(DIR, FRAMES_DIR, "%s"%(img_name))
        xc.create_dir(target_dir)
        for d in dirs:
            source_dir = os.path.join(DIR, d);
            for suffix in SUFFIXES:
                shutil.copyfile(
                    os.path.join(source_dir, "%s%s"%(img_name, suffix)),
                    os.path.join(target_dir, "mattes", "%s_%s_%s"%(d, img_name, suffix))
                )



