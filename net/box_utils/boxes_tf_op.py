# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .iou_tf_op import iou_calculate

#target box和labels 是相关联的 以 box_refs 为参考项, 删除 iou不满足 threshold 条件的 target box
def box_filter_with_iou_for_preprocess(box_refs, target_boxs, labels, threshold=0.1, assign_negative = False):

    #计算iou分数
    scores = iou_calculate(box_refs, target_boxs)

    negative_label = tf.zeros(tf.shape(labels),dtype=labels.dtype) - 1

    #大于0.1阈值的box都会被选择进去
    mask = scores > threshold

    # if assign_negative == True:
    #
    #     #mask为 True的地方 label 为false的地方 设置为 -1
    #     new_labels = tf.where(mask, labels, negative_label)
    #     new_bboxes = target_boxs
    #
    # else:
    #     #把不满足条件的box和对应的label删除掉
    #     mask = tf.reshape(mask,[-1])
    #     new_labels = tf.boolean_mask(labels, mask)
    #     new_bboxes = tf.boolean_mask(target_boxs, mask)

    new_labels = labels
    new_bboxes = target_boxs

    return new_labels, new_bboxes

#将box超过区域的部分进行截断处理
def clip_boxes_to_img_boundaries(boxes_info):

    ymin,xmin,ymax,xman = tf.split(boxes_info, 4, axis=1)

    max_xmin = tf.maximum(0.0, xmin)
    max_ymin = tf.maximum(0.0, ymin)

    min_xmax = tf.minimum(1.0, xman)
    min_ymax = tf.minimum(1.0, ymax)

    new_box_info = tf.concat([max_ymin, max_xmin, min_ymax, min_xmax], axis=1)

    return new_box_info

def decode_boxes(encode_boxes, reference_boxes, scale_factors=None):
    '''
    :param encode_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale
    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 4]
    '''

    t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1)

    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return tf.transpose(tf.stack([predict_ymin, predict_xmin,
                                  predict_ymax, predict_xmax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes:  ground truth box
    :param reference_boxes: anchor
    :return: encode_boxes [-1, 4]
    '''

    ymin, xmin, ymax, xmax = tf.unstack(unencode_boxes, axis=1)

    reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

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

    # t_w = tf.log(w / reference_w)
    # t_h = tf.log(h / reference_h)
    #否则会出现nan的情况
    t_w = tf.log(tf.clip_by_value(w / reference_w, 1e-8, 1000.0))
    t_h = tf.log(tf.clip_by_value(h / reference_h, 1e-8, 1000.0))

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w]))




