# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import glob
import numpy as np

R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def read_image_and_whiten(image_name, image_root, resize = 300, center_image = [_B_MEAN, _G_MEAN, R_MEAN] ):

    image_abs_name = os.path.join(image_root, "VOC2007", "JPEGImages", (image_name.decode('utf-8')))
    image = cv2.imread(image_abs_name)

    if image is None:

        image_abs_name = os.path.join(image_root, "VOC2012", "JPEGImages", (image_name.decode('utf-8')))
        image = cv2.imread(image_abs_name)

        if image is None:

            raise Exception("error read_image_and_whiten")


    image_whiten = image - center_image

    return cv2.resize(image, (resize, resize)), image_whiten


def read_image_with_dir(image_dir, resize = 300 , center_image = [R_MEAN ,_G_MEAN,  _B_MEAN] ):


    image_list = glob.glob(os.path.join(image_dir , "*.jp*g"))

    image_original_array = []
    image_whiten_array = []

    for image_path in image_list:

        image = cv2.imread(image_path)
        image_whiten = cv2.cvtColor(cv2.resize(image,(resize,resize)), code=cv2.COLOR_BGR2RGB) - center_image

        image_original_array.append(cv2.resize(image,(resize,resize)))

        image_whiten_array.append(image_whiten)

    val_original = np.stack(image_original_array, axis=0)
    val_whiten = np.stack(image_whiten_array, axis=0)

    return val_original.astype(np.float32) , val_whiten.astype(np.float32)
