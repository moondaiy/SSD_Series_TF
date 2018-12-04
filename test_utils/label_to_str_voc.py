# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

NAME_LABEL_MAP = {
    'back_ground': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}


def get_key(dict, value):

    return [k for k, v in dict.items() if v == value]

def convert_label_to_str(label):

    # label_value = np.argmax(label)

    label_name = get_key(NAME_LABEL_MAP, label)

    return label_name

