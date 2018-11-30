# -*- coding: utf-8 -*-
import tensorflow as tf
from net.libs import getLayer
from net.box_utils import anchor_np_op
import numpy as np
from net.loss import ssd_loss
from net.box_utils.boxes_tf_op import decode_boxes


class SSD_VGG(object):

    def __init__(self, size):

        self.base_size = size

    def build_feature_layer(self, inputs, is_training, size, extra_info = None):

        if size == 300:

            out = self.vgg_feature_extract_300(inputs, is_training)

        # elif size == 512:
        #
        #     pass

        else:

            raise Exception("Only support vgg size is 300")

        return out


    def construct_block_layer(self,net, convolution_repeat_number, channel_out, kernel_size, slide, scope):

        current_block_scope_name = scope


        net = self.construct_repeat_convolution_layer(convolution_repeat_number, net, channel_out, kernel_size, slide, current_block_scope_name)
        net = self.construct_max_pool_layer(net, current_block_scope_name)

        return net


    def construct_max_pool_layer(self, net, scope, slide_h = 2, slide_w = 2):

        current_scope_name = scope + "_max_pool"
        kernel_h = 2
        kernel_w = 2
        slide_h = slide_h
        slide_w = slide_w

        with tf.variable_scope(current_scope_name):

            net = getLayer.max_pooling_layer(net, kernel_h, kernel_w, slide_h, slide_w)

        return net


    def construct_repeat_convolution_layer(self, net, repeat_number, channel_out, kernel_size, slide, scope, padding = getLayer.DEFAULT_PADDING):

        kerne_h = kernel_size
        kerne_w = kernel_size
        slide_h = slide
        slide_w = slide

        for i in range(repeat_number):

            current_scope_name = scope + "_conv_" + str(i)

            net = getLayer.convolution_layer(net, channel_out, kerne_h, kerne_w, slide_h, slide_w, current_scope_name,
                                       activationFn = tf.nn.relu, padding=padding)

        return net

    def construct_atrous_convolution_layer(self, net, channel_out, kernel_size,rate,scope):

        kerne_h = kernel_size

        kerne_w = kernel_size

        current_scope_name = scope + "_atrous_conv"

        net = getLayer.atrous_conv_2d_layer(net, channel_out, kerne_h, kerne_w, rate, current_scope_name, activationFn = tf.nn.relu)

        return net


    def vgg_feature_extract_300(self, inputs, is_training, drop_out = 0.5):

        end_points = {}

        with tf.variable_scope('vgg_feature_extract_300'):

            net = self.construct_repeat_convolution_layer(inputs, 2, 64, 3, 1, "block_1")
            end_points["block_1"] = net
            net = self.construct_max_pool_layer(net, "block_1")

            net = self.construct_repeat_convolution_layer(net, 2, 128, 3, 1, "block_2")
            end_points["block_2"] = net
            net = self.construct_max_pool_layer(net, "block_2")

            net = self.construct_repeat_convolution_layer(net, 3, 256, 3, 1, "block_3")
            end_points["block_3"] = net
            net = self.construct_max_pool_layer(net, "block_3")

            net = self.construct_repeat_convolution_layer(net, 3, 512, 3, 1, "block_4")
            end_points["block_4"] = net
            net = self.construct_max_pool_layer(net, "block_4")

            net = self.construct_repeat_convolution_layer(net, 3, 512, 3, 1, "block_5")
            end_points["block_5"] = net
            net = self.construct_max_pool_layer(net, "block_5", slide_h = 1, slide_w = 1)

            net = self.construct_atrous_convolution_layer(net, 1024, 3, 6, "block_6")
            end_points["block_6"] = net
            net = getLayer.dropout_layer(net, drop_out, is_training)

            net = self.construct_repeat_convolution_layer(net, 1, 1024, 1, 1, "block_7")
            end_points['block_7'] = net
            net = getLayer.dropout_layer(net, drop_out, is_training)

            net = self.construct_repeat_convolution_layer(net, 1, 256, 1, 1, "block_8_1X1")
            net = getLayer.pad_2d(net, pad=(1,1),name_scope="block_8_pad")
            net = self.construct_repeat_convolution_layer(net, 1, 512, 3, 2, "block_8_3X3", padding="VALID")
            end_points["block_8"] = net

            net = self.construct_repeat_convolution_layer(net, 1, 128, 1, 1, "block_9_1X1")
            net = getLayer.pad_2d(net, pad=(1,1),name_scope="block_9_pad")
            net = self.construct_repeat_convolution_layer(net, 1, 256, 3, 2, "block_9_3X3", padding="VALID")
            end_points["block_9"] = net

            net = self.construct_repeat_convolution_layer(net, 1, 128, 1, 1, "block_10_1X1")
            net = self.construct_repeat_convolution_layer(net, 1, 256, 3, 1, "block_10_3X3", padding="VALID")
            end_points["block_10"] = net

            net = self.construct_repeat_convolution_layer(net, 1, 128, 1, 1, "block_11_1X1")
            net = self.construct_repeat_convolution_layer(net, 1, 256, 3, 1, "block_11_3X3", padding="VALID")
            end_points["block_11"] = net

            return end_points

    def build_anchors_with_anchor_infos(self, anchor_infos,image_size):

        base_anchor_sizes = anchor_infos["anchor_size"]
        anchor_ratios = anchor_infos["anchor_ratios"]
        feat_shapes = anchor_infos["feature_shape"]
        anchor_strides = anchor_infos["anchor_strides"]
        anchor_offset = anchor_infos["anchor_offset"]

        all_anchors = anchor_np_op.make_anchors_for_all_layer(image_size, image_size, base_anchor_sizes, anchor_ratios, anchor_strides, feat_shapes, anchor_offset=anchor_offset)

        #返回的是numpy类型数据
        return np.concatenate(all_anchors, axis=0)

    def build_valid_feature_layer(self, base_feature, valid_layer_list, base_net_size):

        valid_feature_list = []

        for feature_name in valid_layer_list:

            if feature_name in base_feature.keys():

                valid_feature_list.append(base_feature[feature_name])

            else:

                raise Exception("There is a feature layer not in base feature ")

        return valid_feature_list



    def single_multibox_layer(self,inputs, class_number, anchor_base_size, anchor_ratio, normalization_factor , current_feature_size_h, current_feature_size_w, scope_name):

        nets = inputs

        if normalization_factor != 0:

            l2_normalization_name = scope_name + "_l2_normalization"

            nets = getLayer.normalization_l2_layer(nets, scale_factor=normalization_factor, scope_name=l2_normalization_name)

        num_anchors = len(anchor_base_size) + len(anchor_ratio)


        #预测box的channel信息
        num_loc_pred_number = num_anchors * 4
        num_cls_pred_number = num_anchors * class_number

        #方便loss计算 进行的tensor shape操作
        reshape_loc_tesnor = [-1, current_feature_size_h * current_feature_size_w * num_anchors, 4]
        reshape_cls_tesnor = [-1, current_feature_size_h * current_feature_size_w * num_anchors, class_number]

        #loc   信息获得
        loc_pred_name = scope_name + "_location"
        loc_pred = getLayer.convolution_layer(nets, num_loc_pred_number, 3, 3, 1, 1, loc_pred_name)
        loc_pred = tf.reshape(loc_pred, shape=reshape_loc_tesnor)
        # loc_pred = tf.reshape(loc_pred, shape=[-1, -1, 4])

        #class 信息获得
        cls_pred_name = scope_name + "_classfication"
        cls_pred = getLayer.convolution_layer(nets, num_cls_pred_number, 3, 3, 1, 1, cls_pred_name)
        cls_pred = tf.reshape(cls_pred, shape=reshape_cls_tesnor)

        return loc_pred, cls_pred

    def build_multibox_layer(self, input_list, input_name_list, class_number, anchor_base_size_list, anchor_ratio_list, normalization_list, feature_size_list):

        loc_pre_list = []
        cls_pre_list = []

        for i in range(len(input_name_list)):

            current_input            = input_list[i]
            current_layer_name       = input_name_list[i]
            current_anchor_base_size = anchor_base_size_list[i]
            current_anchor_ratio     = anchor_ratio_list[i]
            current_normalization_factor    = normalization_list[i]
            current_feature_size_h     = feature_size_list[i][0]
            current_feature_size_w     = feature_size_list[i][1]

            current_scope_name = current_layer_name + "_multibox_layer"

            current_loc_pred, current_cls_pred = self.single_multibox_layer(current_input, class_number, current_anchor_base_size, current_anchor_ratio, current_normalization_factor, current_feature_size_h, current_feature_size_w, current_scope_name)

            loc_pre_list.append(current_loc_pred)
            cls_pre_list.append(current_cls_pred)

        #所有的合并起来
        loc_tensor = tf.concat(loc_pre_list,axis=1)
        cls_tensor = tf.concat(cls_pre_list,axis=1)

        logistic_tensor = tf.concat([cls_tensor, loc_tensor], axis=2)

        return logistic_tensor

    def build_ssd_loss(self,logitic_tensor, encode_gt_tensor, total_anchor_number, neg_pos_ratio = 3.0, min_negative_number = 0, alpha = 1.0):

        total_localization_loss, total_classification_loss = ssd_loss.build_ssd_loss(logitic_tensor, encode_gt_tensor, total_anchor_number, neg_pos_ratio, min_negative_number, alpha)

        total_loss = total_classification_loss + alpha * total_localization_loss

        return total_localization_loss, total_classification_loss, total_loss

    def build_ssd_net_out(self, logistic_tensor):

        logistic_label_tensor = logistic_tensor[:,:,0:21]
        logistic_bbox_tensor  = logistic_tensor[:,:,21:25]

        #soft max 操作
        soft_max_logistc_tensor = tf.nn.softmax(logistic_label_tensor)

        predict_tensor = tf.concat([soft_max_logistc_tensor, logistic_bbox_tensor], axis=2)

        return predict_tensor











