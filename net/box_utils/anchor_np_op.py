# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

#paper中Smax - Smin = 0.9 - 0.2 这个而参数可以自行更改
#Sk = Smin + (Smax - Smin)/(m - 1)  * (k - 1) k 属于[1, m] m= 5 第一层的anchor设置不符合该公式
#第一层conv4_3 上的anchors尺寸 Smin/2 = 0.1  (假设是300 图像大小 则 Base anchor size = 30)
#base_anchor_sizes = [(30 60), (60 111), (111 162), (162 213), (213, 264), (264 315)] 这些值得可以根据自己需要进行更改
def make_one_layer_base_anchor(image_shape, base_anchor_sizes, anchor_ratios, anchor_strides, feature_height, feathre_width, offset=0.5,style="coord"):
    """
    :param image_shape:       图像大小
    :param base_anchor_sizes: 基础anchor尺寸 相对于原图的
    :param anchor_ratios:     anchor的比例关系
    :param anchor_strides:    anchor的间隔(在原图上)
    :param feature_height:   当前feature map 高度
    :param feathre_width:    当前feature map 宽度
    :param offset:           anchor 中心店相对于grid的偏移
    :return:                 根据当前设置 返回的 anchor
    """
    current_layer_anchor_w_size = []
    current_layer_anchor_h_size = []

    image_width  = image_shape[0]
    image_height = image_shape[1]

    normal_anchor_base_size = base_anchor_sizes[0]
    special_anchor_base_size = base_anchor_sizes[1]

    #这个anchor表示了Sk和Sk+1
    speical_anchor_size = math.sqrt(normal_anchor_base_size * special_anchor_base_size)

    #根据论文要求 每层都要有一个 Sk = sqrt(Sk * Sk+1) 而且 ratio = 1的先验框
    current_layer_anchor_w_size.append(speical_anchor_size / image_width)
    current_layer_anchor_h_size.append(speical_anchor_size / image_height)

    #加入一个 base anchor size 为 normal_anchor_base_size 且 ar = 1的 anchor box
    current_layer_anchor_w_size.append(normal_anchor_base_size / image_width)
    current_layer_anchor_h_size.append(normal_anchor_base_size / image_height)

    #计算笔筒比例下的 anchor size
    for ratio in anchor_ratios:

        w = normal_anchor_base_size * math.sqrt(ratio) / image_width
        h = normal_anchor_base_size / math.sqrt(ratio) / image_height

        current_layer_anchor_w_size.append(w)
        current_layer_anchor_h_size.append(h)

    current_layer_anchor_w_size = np.expand_dims(np.array(current_layer_anchor_w_size), axis= 0)
    current_layer_anchor_h_size = np.expand_dims(np.array(current_layer_anchor_h_size), axis= 0)


    #计算anchor中心坐标点相对于图像(不是feature map)的偏移
    x_centers = (np.arange(0 , int(feathre_width),  1 , dtype=np.float32) + offset )* anchor_strides / image_width
    y_centers = (np.arange(0 , int(feature_height), 1 , dtype=np.float32) + offset )* anchor_strides / image_height

    #组成二维meshgrid形式
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)


    #利用传播特性进行加法方便计算
    x_centers = np.expand_dims(x_centers,axis=-1)
    y_centers = np.expand_dims(y_centers,axis=-1)

    current_layer_anchor_w_size = np.zeros_like(x_centers) + current_layer_anchor_w_size
    current_layer_anchor_h_size = np.zeros_like(y_centers) + current_layer_anchor_h_size

    ws = np.reshape(current_layer_anchor_w_size, [-1, 1])
    hs = np.reshape(current_layer_anchor_h_size, [-1, 1])

    x  = np.reshape(x_centers + np.zeros_like(current_layer_anchor_w_size), [-1, 1])
    y  = np.reshape(y_centers + np.zeros_like(current_layer_anchor_h_size), [-1, 1])

    if style == "coord":

        ymin = y - hs/2
        xmin = x - ws/2
        ymax = y + hs / 2
        xmax = x + ws/2

        return np.concatenate((ymin,xmin,ymax,xmax),axis=1)

    elif style == "center":

        return np.concatenate((y, x, hs, ws), axis=1)

    else:

        raise Exception("make_one_layer_base_anchor support style : coord ,center")

def make_anchors_for_all_layer(image_height,image_width, base_anchor_sizes, anchor_ratios, anchor_strides, feat_shapes, anchor_offset=0.5):
    """
    :param image_height: 图像高度
    :param image_width:  图像宽度
    :param base_anchor_sizes:  list类型 base_size , special_size
    :param anchor_ratios: list类型 anchor的宽高比例 1:1 和 special不在其中表示 作为默认每一层均需要实现
    :param anchor_strides:在原图上, 相邻两个anchor中心的间距
    :param feat_shapes: 每个特征层的分辨率
    :param anchor_offset: anchor中心相对于grid的位移
    :return: list 每层的anchor 数量
    """
    layer_number = len(base_anchor_sizes)
    total_anchors = []
    total_number = 0

    for i in range(layer_number):

        anchors = make_one_layer_base_anchor((image_height,image_width), base_anchor_sizes[i],
                                   anchor_ratios[i],
                                   anchor_strides[i],
                                   feat_shapes[i][0],
                                   feat_shapes[i][0],
                                   offset=anchor_offset)

        total_number += len(anchors)

        total_anchors.append(anchors)

    # total_anchors = np.concatenate(total_anchors,axis=0)

    return total_anchors, total_number


def display_anchor_box(anchor_boxs, origianl_size , image, display_layer, display_number):

    import cv2

    image_height ,image_width,image_channl = image.shape
    count_number = 0

    original_image_height = origianl_size
    origianl_image_width  = origianl_size

    for x,y,w,h in anchor_boxs[display_layer]:

        start_x = int(x * origianl_image_width) if int(x * origianl_image_width) >=0 else 0
        start_y = int(y * original_image_height) if int(y * original_image_height) >= 0 else 0
        end_x = int((w * origianl_image_width )) if int(x * origianl_image_width) < origianl_image_width else origianl_image_width
        end_y = int((h * original_image_height)) if int(y * original_image_height)<  origianl_image_width else origianl_image_width



        cv2.rectangle(image, (50, 50), (350, 350), thickness=1, color=(0, 0, 255))

        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), thickness=1, color=(255,255,255))

        count_number += 1

        if count_number < display_number:

            continue

        else:
            cv2.imshow("1", image)
            cv2.waitKey(0)
            image -= image
            count_number = 0
        # cv2.destroyAllWindows()

