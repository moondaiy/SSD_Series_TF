# -*- coding: utf-8 -*-
import tensorflow as tf
from net.libs import getLayer
from net.box_utils import anchor_np_op
import numpy as np
from net.loss import ssd_loss
import net.libs.dense_net_api as dense_api

#DSOD 的实现方式
class SSD_DenseNet(object):

    #对应原始paper中介绍 stem层
    #dense block 层

    def __init__(self, input_image_size):

        self.input_image_size = input_image_size

    def dense_feature_extract_300(self, inputs, is_training, extra_info=None):

        # self.is_training = is_training

        end_points = {}
        bias = False

        #没办法显卡空间不够了. 原始papaer是 48
        growth_rate = 48

        #不使用drop out技术
        drop_ratio = 1.0


        group_number = 4


        # return self.vgg_feature_extract_300(inputs, is_training)

        with tf.variable_scope('dsod_feature_extract_300'):

            #net 128×75×75
            net = self.build_stem_block_for_dsod(inputs, is_training, group_number, "stem")

            #net 416×38×38
            net = dense_api.build_dense_block(net, 6, growth_rate, 3, 3, 1, 1, is_training, drop_ratio, True, bias , group_number, "dense_block_1_block")
            net, out_put_no_pooling = dense_api.build_transition_layer(net , 2, 2, 1.0, is_training, drop_ratio, bias, group_number, name="dense_block_1_transition")


            net = dense_api.build_dense_block(net, 8, growth_rate, 3, 3, 1, 1, is_training, drop_ratio, True, bias, group_number, "dense_block_2_block")
            net, out_put_no_pooling = dense_api.build_transition_layer(net, 2, 2, 1.0, is_training, drop_ratio, bias, group_number, name="dense_block_2_transition")

            # first 800×38×38
            end_points["1"] = out_put_no_pooling

            net = dense_api.build_dense_block(net, 8, growth_rate, 3, 3, 1, 1, is_training, drop_ratio, True, bias, group_number, "dense_block_3_block")
            net = dense_api.build_transition_w_o_layer(net , is_training, drop_ratio, bias=bias, group = group_number, name="dense_block_3_transition_w_o_layer")

            net = dense_api.build_dense_block(net, 8, growth_rate, 3, 3, 1, 1, is_training, drop_ratio, True, bias, group_number, "dense_block_4_block")
            net = dense_api.build_transition_w_o_layer(net , is_training, 1.0, bias=bias, group = group_number, name="dense_block_4_transition_w_o_layer")

            # backbone_out 1568×19×19
            # backbone_out = net

            second_pre_right = dense_api.build_composite_brc_layer(net, 256, 1, 1, 1, 1, is_training, drop_ratio, bias=bias , group = group_number, name ="second_feature_right_process")
            second_pre_left  = self.pre_process_left_feature(out_put_no_pooling, 256, is_training, drop_ratio, bias=bias , group = group_number, name="second_feature_left_process")
            second_feature = tf.concat([second_pre_left, second_pre_right],axis=3)
            end_points["2"] = second_feature

            third_pre_right = self.pre_process_right_feature(second_feature, 256, is_training, drop_ratio, bias=bias , group = group_number, name="third_feature_left_process")
            third_pre_left  = self.pre_process_left_feature( second_feature, 256, is_training, drop_ratio, bias=bias , group = group_number, name="third_feature_left_process")
            third_feature = tf.concat([third_pre_left, third_pre_right],axis=3)
            end_points["3"] = third_feature


            forth_pre_right = self.pre_process_right_feature(third_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="forth_feature_left_process")
            forth_pre_left  = self.pre_process_left_feature( third_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="forth_feature_left_process")
            forth_feature = tf.concat([forth_pre_left, forth_pre_right],axis=3)
            end_points["4"] = forth_feature


            fifth_pre_right = self.pre_process_right_feature(forth_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="fifth_feature_left_process")
            fifth_pre_left  = self.pre_process_left_feature( forth_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="fifth_feature_left_process")
            fifth_feature = tf.concat([fifth_pre_left, fifth_pre_right],axis=3)
            end_points["5"] = fifth_feature


            sixth_pre_right = self.pre_process_right_feature(fifth_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="sixth_feature_left_process" , padding="VALID")
            sixth_pre_left  = self.pre_process_left_feature( fifth_feature, 128, is_training, drop_ratio, bias=bias , group = group_number, name="sixth_feature_left_process" , stride  = 2, padding="VALID")
            sixth_feature = tf.concat([sixth_pre_left, sixth_pre_right],axis=3)
            end_points["6"] = sixth_feature

            return end_points

    def pre_process_left_feature(self, inputs, output_channel_number, is_training, keep_prob, bias , group , name, stride = 1, padding = "SAME"):

        with tf.variable_scope(name) as scope:

            net = getLayer.max_pooling_layer(inputs)

            net = dense_api.build_composite_brc_layer(net, output_channel_number, 1, 1, stride, stride, is_training, keep_prob, bias , group , name ="bn_relu_conv", padding = padding)

            return net

    #参考 http://ethereon.github.io/netscope/#/gist/b17d01f3131e2a60f9057b5d3eb9e04d 的左右分支处理
    def pre_process_right_feature(self, inputs, output_channel_number, is_training, keep_prob, bias, group, name, padding = "SAME"):

        with tf.variable_scope(name) as scope:

            net = dense_api.build_composite_brc_layer(inputs, output_channel_number, 1, 1, 1, 1, is_training, keep_prob, bias, group, name="bn_relu_conv_1", padding = padding)
            net = dense_api.build_composite_brc_layer(net,    output_channel_number, 3, 3, 2, 2, is_training, keep_prob, bias, group, name="bn_relu_conv_2", padding = padding)

            return net


    def build_valid_feature_layer(self, base_feature, valid_layer_list, base_net_size):

        valid_feature_list = []

        for feature_name in valid_layer_list:

            if feature_name in base_feature.keys():

                valid_feature_list.append(base_feature[feature_name])

            else:

                raise Exception("There is a feature layer not in base feature ")

        return valid_feature_list


    def build_stem_block_for_dsod(self, inputs , is_training , group = 4, name = "stem"):

        #根据paper中的说明 stem block 这样组成的 但是注意的是 bias 是false的
        # keep_prop = 1.0 则说明不需要drop out 操作
        with tf.variable_scope(name) as scope:

            inputs = dense_api.build_composite_cbr_layer(inputs, 64,  3, 3, 2, 2, is_training, keep_prob = 1.0, bias= False, group = group, name ="conv_bn_relu_1")

            inputs = dense_api.build_composite_cbr_layer(inputs, 64,  3, 3, 1, 1, is_training, keep_prob = 1.0, bias= False, group = group,  name ="conv_bn_relu_2")

            inputs = dense_api.build_composite_cbr_layer(inputs, 128, 3, 3, 1, 1, is_training, keep_prob = 1.0, bias= False, group = group,  name ="conv_bn_relu_3")

            inputs = getLayer.max_pooling_layer(inputs, kernel_h = 2, kernel_w = 2, slide_h = 2, slide_w = 2)

            return inputs


    #对外接口 提取feature layer的主干网络
    def build_feature_layer(self, inputs, is_training, size):

        if size == 300:

            out = self.dense_feature_extract_300(inputs, is_training)

        # elif size == 512:
        #
        #     pass

        else:

            raise Exception("Only support vgg size is 300")

        return out


    def build_multibox_layer(self, input_list, input_name_list, class_number, anchor_base_size_list, anchor_ratio_list, normalization_list, feature_size_list, is_train):

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

            current_loc_pred, current_cls_pred = self.single_multibox_layer(current_input, class_number, current_anchor_base_size, current_anchor_ratio, current_normalization_factor, current_feature_size_h, current_feature_size_w, is_train, current_scope_name)

            loc_pre_list.append(current_loc_pred)
            cls_pre_list.append(current_cls_pred)

        #所有的合并起来
        loc_tensor = tf.concat(loc_pre_list,axis=1)
        cls_tensor = tf.concat(cls_pre_list,axis=1)

        logistic_tensor = tf.concat([cls_tensor, loc_tensor], axis=2)

        return logistic_tensor


    def single_multibox_layer(self,inputs, class_number, anchor_base_size, anchor_ratio, normalization_factor , current_feature_size_h, current_feature_size_w, is_train, scope_name):

        nets = inputs
        bias = False
        # is_training = True

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
        loc_pred = getLayer.convolution_layer(nets, num_loc_pred_number, 3, 3, 1, 1, loc_pred_name, biased=bias)
        # loc_pred = getLayer.group_normalization_layer(loc_pred , num_anchors , loc_pred_name + "bn") #num_anchors
        loc_pred = getLayer.normalization_layer(loc_pred, is_train, loc_pred_name + "bn", "bn", num_anchors, True)
        loc_pred = tf.reshape(loc_pred, shape=reshape_loc_tesnor)

        #class 信息获得
        cls_pred_name = scope_name + "_classfication"
        cls_pred = getLayer.convolution_layer(nets, num_cls_pred_number, 3, 3, 1, 1, cls_pred_name, biased=bias)
        # cls_pred = getLayer.group_normalization_layer(cls_pred, num_anchors , cls_pred_name + "bn")
        cls_pred = getLayer.normalization_layer(cls_pred, is_train, cls_pred_name + "bn", "bn", num_anchors, True)
        cls_pred = tf.reshape(cls_pred, shape=reshape_cls_tesnor)

        return loc_pred, cls_pred


    def build_ssd_loss(self,logitic_tensor, encode_gt_tensor, total_anchor_number, neg_pos_ratio = 3.0, min_negative_number = 0, alpha = 1.0):

        total_localization_loss, total_classification_loss = ssd_loss.build_ssd_loss_adv(logitic_tensor, encode_gt_tensor, total_anchor_number, neg_pos_ratio, min_negative_number, alpha)

        total_loss = total_classification_loss + alpha * total_localization_loss

        return total_localization_loss, total_classification_loss, total_loss

    def build_ssd_net_out(self, logistic_tensor):

        logistic_label_tensor = logistic_tensor[:,:,0:21]
        logistic_bbox_tensor  = logistic_tensor[:,:,21:25]

        #soft max 操作
        soft_max_logistc_tensor = tf.nn.softmax(logistic_label_tensor)

        predict_tensor = tf.concat([soft_max_logistc_tensor, logistic_bbox_tensor], axis=2)

        return predict_tensor




