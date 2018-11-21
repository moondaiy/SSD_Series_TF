# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def decode_boxes(encode_boxes, reference_boxes, scale_factors=None):
    '''
    :param encode_boxes:  predict
    :param reference_boxes: anchor
    :return: encode_boxes [-1, 4]
    '''

    # t_ycenter, t_xcenter, t_h, t_w = np.unstack(encode_boxes, axis=1)
    t_ycenter = encode_boxes[:,0]
    t_xcenter = encode_boxes[:,1]
    t_h       = encode_boxes[:,2]
    t_w       = encode_boxes[:,3]

    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_ymin = reference_boxes[:,0]
    reference_xmin = reference_boxes[:,1]
    reference_ymax = reference_boxes[:,2]
    reference_xmax = reference_boxes[:,3]

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = np.exp(t_w) * reference_w
    predict_h = np.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return np.transpose(np.stack([predict_ymin, predict_xmin,
                                  predict_ymax, predict_xmax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes:  ground truth box
    :param reference_boxes: anchor
    :return: encode_boxes [-1, 4]
    '''

    ymin = unencode_boxes[:,0]
    xmin = unencode_boxes[:,1]
    ymax = unencode_boxes[:,2]
    xmax = unencode_boxes[:,3]


    reference_ymin = reference_boxes[:,0]
    reference_xmin = reference_boxes[:,1]
    reference_ymax = reference_boxes[:,2]
    reference_xmax = reference_boxes[:,3]

    x_center = (xmin + xmax) / 2.
    y_center = (ymin + ymax) / 2.
    w = xmax - xmin
    h = ymax - ymin

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    reference_w += 1e-8
    reference_h += 1e-8
    w += 1e-8
    h += 1e-8  # to avoid NaN in division and log below

    t_xcenter = (x_center - reference_xcenter) / reference_w
    t_ycenter = (y_center - reference_ycenter) / reference_h
    t_w = np.log(w / reference_w)
    t_h = np.log(h / reference_h)

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return np.transpose(np.stack([t_ycenter, t_xcenter, t_h, t_w]))



def anchor_box_convert(anchors, target_type = "coord"):

    if target_type == "coord":
        #ymin xmin ymax xmax
        y_center = anchors[:,0]
        x_center = anchors[:,1]
        h_szie   = anchors[:,2]
        w_size   = anchors[:,3]

        ymin = y_center - h_szie/2
        xmin = x_center - w_size/2
        ymax = y_center + h_szie/2
        xmax = x_center + w_size/2


        new_anchor = np.stack([ymin, xmin, ymax, xmax],axis=1)

    else:

        raise Exception("anchor_box_convert support center -> coord")



    return new_anchor