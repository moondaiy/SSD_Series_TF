# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from test_utils.label_to_str_voc import convert_label_to_str


def render_boxs_info_for_display(image, net_out, select_index, net_score, image_size, label_out = None):


    valid_box = net_out[select_index]
    valid_score = net_score[select_index]

    for index, value in enumerate(select_index):

        if net_score[index] > 0.5 and value == True:
        # if value == True:

            valid_box = net_out[index]
            valid_score = net_score[index]

            print("current box info is " + str(valid_box))
            print("current box scores is " + str(valid_score))

            if label_out is not None :
                print("current label is %s"%(convert_label_to_str(label_out[index])))


            ymin = int(valid_box[0] * image_size)
            xmin = int(valid_box[1] * image_size)
            ymax = int(valid_box[2] * image_size)
            xmax = int(valid_box[3] * image_size)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), thickness=1,color=(0,0,255))

    return image

def render_rectangle_box(image, box, colour = (255, 255, 255), offset = 0, thickness = 1):
    """
    :param image: 需要显示的图片
    :param box:   box信息
    :param colour: 颜色信息
    :param offset: box偏移
    :param thickness: 线条宽度
    :return:
    """

    height,width, channel = image.shape

    y_start = int(height * box[0]) + offset
    x_start = int(width  * box[1]) + offset

    y_end = int(height * box[2]) + offset
    x_end = int(width  * box[3]) + offset

    image = cv2.rectangle(image,(x_start,y_start), (x_end,y_end), color=colour, thickness= thickness)

    return image