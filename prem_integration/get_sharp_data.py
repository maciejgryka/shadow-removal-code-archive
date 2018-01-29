import os

import numpy as np
from scipy.ndimage.filters import sobel
import matplotlib.pyplot as plt

import experiment_common as xc


TRAINING_DATA_DIR = "C:\\Work\\research\\shadow_removal\\experiments\\training_images"


if __name__ == "__main__":
    img_names = xc.separate_image_names(
        xc.get_file_names(TRAINING_DATA_DIR, xc.is_shad)
    )

    plt.gray()
    for img_name in img_names[:20]:
        shad = xc.imread_channel(os.path.join(TRAINING_DATA_DIR, img_name+"_shad.png"))
        noshad = xc.imread_channel(os.path.join(TRAINING_DATA_DIR, img_name+"_noshad.png"))
        matte = shad/noshad
        grad = abs(sobel(matte))
        # plt.imshow(grad > 0.8, interpolation="nearest")
        if grad > 0.8
        print img_name

    plt.show()
