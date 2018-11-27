# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from .iou_tf_op import iou_calculate
from .boxes_tf_op import encode_boxes


#转换anchor y_centers , x_centers ,height , width -> ymin xmin ymax xmax
def convert_anchor(anchors):

    anchors_tensor = tf.constant(value=anchors)

    ymin = anchors_tensor[:, 0] - anchors_tensor[:, 2] / 2
    xmin = anchors_tensor[:, 1] - anchors_tensor[:, 3] / 2
    ymax = anchors_tensor[:, 0] + anchors_tensor[:, 2] / 2
    xmax = anchors_tensor[:, 1] + anchors_tensor[:, 3] / 2

    return tf.stack([ymin, xmin, ymax, xmax],axis=1)


def calculate_score(ancor_tensor, ground_truth_boxs):

    #anchor = 300 gt = 4 则 score 300 * 4
    score = iou_calculate(ancor_tensor, ground_truth_boxs)

    return score


#将输入的ground truth box 和 label进行encoding成 训练可用的方式
#每次输入的是一个图像信息
def  generate_train_encoding_labels_tf_operation(anchors_tensor, total_number, unencoded_ground_truth_bbox_tensor, unencoded_ground_truth_label_tensor, scale_factors , class_number = 21, pos_iou_threshold = 0.5, negtive_id = 0):
    """
    :param anchors:
    :param ground_truth_boxs:
    :param ground_truth_labels:
    :param pos_iou_threshold:
    :return:
    """
    score_list = []
    anchor_index = 0

    #保存当前anchor的label(one_hot 形式的状态)
    encoded_ground_truth_label_shape = (total_number, class_number)

    #保存当前anchor需要encode的box信息
    encoded_ground_truth_bbox_shape  = (total_number, 4)

    #保存当前anchor的得分
    encodeed_scores_shape = (total_number, 1)

    #初始化这些信息
    # encoded_ground_truth_label_tensor = tf.zeros(encoded_ground_truth_label_shape)
    #初始化时候全部赋值成负样本形式
    encoded_ground_truth_label_tensor = tf.tile(tf.expand_dims(tf.sparse_to_dense([negtive_id], [class_number], 1.0, 0.0), axis=0), multiples=[total_number, 1])

    encoded_ground_truth_bbox_tensor =  tf.zeros(encoded_ground_truth_bbox_shape)

    unencode_recoard_ground_truth_bbox_tensor =  tf.zeros(encoded_ground_truth_bbox_shape) #记录原始box的信息

    encodeed_scores_tensor = tf.zeros(encodeed_scores_shape)

    def body(i, anchors_tensor, unencoded_ground_truth_bbox_tensor, unencoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encoded_ground_truth_label_tensor, encodeed_scores_tensor, unencode_recoard_ground_truth_bbox_tensor):

        current_label = tf.one_hot(tf.cast(unencoded_ground_truth_label_tensor[i], dtype=tf.uint8), class_number, on_value=1.0)

        current_score = calculate_score(anchors_tensor, tf.reshape(unencoded_ground_truth_bbox_tensor[i, :], shape=(-1, 4)))

        #当前score 分数 大于 pos_iou_threshold 且 大于上一次的iou分数,第一个标准 是选择那些与gt iou > 0.5的anchor 作为当前Image 训练的正样本.
        #但是遇到小的物品.则会发生 没有正样本的问题.因此同时也要选择 当前gt 和anchor 中 iou 最大的那个
        #正样本选择第一条件
        first_condition = tf.squeeze(tf.logical_and(current_score > pos_iou_threshold, current_score > encodeed_scores_tensor))

        #获得当前gt 与 anchor iou最大的一个 保证一个图片中只要存在 gt box 均可以被选中
        #正样本选择第二条件
        second_condition = tf.argmax(current_score,axis=0)
        second_condition = tf.squeeze(tf.cast(tf.one_hot(second_condition, total_number, on_value = 1),dtype=tf.bool))

        #只要满足2个条件中的一个都可以被选中,作为正样本, 但是实际上 这样选择的方式 正样本灰常少.... 在loss设计上需要进行考虑.如何做到高效的训练. focus loss?
        update_condition = tf.logical_or(first_condition, second_condition)

        #更新score
        encodeed_scores_tensor = tf.where(update_condition, current_score, encodeed_scores_tensor)

        #更新label
        encoded_ground_truth_label_tensor = tf.where(update_condition, tf.tile(tf.expand_dims(current_label, axis=0), multiples = [total_number, 1]), encoded_ground_truth_label_tensor)

        #计算encoded box
        current_encoded_box = encode_boxes(tf.reshape(unencoded_ground_truth_bbox_tensor[i, :], shape=(-1, 4)), anchors_tensor, scale_factors)

        #更新 encoded box
        encoded_ground_truth_bbox_tensor = tf.where(update_condition, current_encoded_box, encoded_ground_truth_bbox_tensor)

        #更新原始box 记录信息 ymin xmin ymax xmax
        unencode_recoard_ground_truth_bbox_tensor = tf.where(update_condition, tf.tile(tf.reshape(unencoded_ground_truth_bbox_tensor[i, :], shape=[-1, 4]), multiples=[total_number, 1]), unencode_recoard_ground_truth_bbox_tensor)

        i = i + 1

        return i, anchors_tensor, unencoded_ground_truth_bbox_tensor, unencoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encoded_ground_truth_label_tensor, encodeed_scores_tensor, unencode_recoard_ground_truth_bbox_tensor


    def condition(i, anchors_tensor, decoded_ground_truth_bbox_tensor, decoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encoded_ground_truth_label_tensor, encodeed_scores_tensor, unencode_ground_truth_bbox_tensor):

        #对gt 中的 label为单位进行遍历,把label中的每一个都和 所有 anchor box进行计算 并更新数据
        r = tf.less(i, tf.shape(decoded_ground_truth_label_tensor))

        return r[0]


    i = 0

    i, anchors_tensor, unencoded_ground_truth_bbox_tensor, unencoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encoded_ground_truth_label_tensor, encodeed_scores_tensor, unencode_recoard_ground_truth_bbox_tensor = \
        tf.while_loop(condition, body, [i, anchors_tensor, unencoded_ground_truth_bbox_tensor, unencoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encoded_ground_truth_label_tensor, encodeed_scores_tensor, unencode_recoard_ground_truth_bbox_tensor])


    #生成一个tensor 用于保存最终的encoded后的 gt
    # 对于某个图片来说是 8732 * [ 当前anchor n_class(21, 0代表背景), 4个anchor所对应的encode box 值, 1 当前anchor的 score ,4 当前 gt box]
    encode_ground_truth_box_label_tensor = tf.concat([encoded_ground_truth_label_tensor, encoded_ground_truth_bbox_tensor, encodeed_scores_tensor, unencode_recoard_ground_truth_bbox_tensor], axis=1)

    return encode_ground_truth_box_label_tensor





























