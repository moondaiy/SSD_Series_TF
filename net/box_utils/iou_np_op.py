# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def iou_calculate(boxes_1, boxes_2):

    #box_1 M * 4
    #box_2 N * 4

    ymin_1, xmin_1, ymax_1, xmax_1 = np.split(boxes_1, 4, axis=1)

    ymin_2 = boxes_2[:,0]
    xmin_2 = boxes_2[:,1]
    ymax_2 = boxes_2[:,2]
    xmax_2 = boxes_2[:,3]

    max_xmin = np.maximum(xmin_1, xmin_2)
    min_xmax = np.minimum(xmax_1, xmax_2)

    max_ymin = np.maximum(ymin_1, ymin_2)
    min_ymax = np.minimum(ymax_1, ymax_2)

    overlap_h = np.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = np.maximum(0., min_xmax - max_xmin)

    #计算交叉面积大小
    overlaps = overlap_h * overlap_w

    #计算各自的面积大小
    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    #计算交叉/并集 结果
    iou = overlaps / (area_1 + area_2 - overlaps)

    #return M * N 的矩阵
    return iou