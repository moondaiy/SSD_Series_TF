# -*- coding: utf-8 -*-
import tensorflow as tf
from net.libs import getLayer
from net.box_utils import anchor_np_op
import numpy as np
from net.loss import ssd_loss

#DSOD的实现方式
class SSD_DenseNet(object):


    def __init__(self):
        pass


    #构造 dense block
    def build_dense_block(self, inputs, layer_number_per_block, growth_rate, kernel_h, kernel_w, slide_h, slide_w, is_training, keep_prob, bc_mode, name ):


        with tf.variable_scope(name):

            for i in range(layer_number_per_block):

                current_layer_name = "single_layer_" + str(i)

                inputs = self.build_one_layer_in_block(inputs, growth_rate, kernel_h , kernel_w , slide_h , slide_w , is_training , keep_prob , bc_mode , current_layer_name)

            output_channel_number = layer_number_per_block * growth_rate

            output = inputs

            return output , output_channel_number

    #构造 transition layer
    def build_transition_layer(self, inputs , pool_stride_h, pool_stride_w, compression_factor, is_training, keep_prob, name="transition_layer"):

        input_channel_number  = inputs.get_shape().as_list()[-1]  # 根据channel得到

        output_channel_number = (int)(input_channel_number * compression_factor) #计算输出 channel 大小

        with tf.variable_scope(name):

            out_put = self.build_composite_layer(inputs, output_channel_number, 1, 1, 1, 1, is_training, keep_prob, "bn_relu_conv")

            out_put = getLayer.dropout_layer(out_put, keep_prob, is_training)

            out_put = getLayer.average_pooling_layer(out_put, kernel_h = 2, kernel_w = 2, slide_h = pool_stride_h, slide_w = pool_stride_w)

            return out_put, output_channel_number


    #构造dense block中的一个layer
    def build_one_layer_in_block(self,inputs, growth_rate, kernel_h = 3, kernel_w = 3, slide_h = 1, slide_w = 1, is_training = True, keep_prob = 0.5, bc_mode = True, name="single_layer"):

        with tf.variable_scope(name):

            #如果使用BC Mode
            if bc_mode == True:

                #BC模式需要加入bottleneck 主要作用是降低维度
                bottleneck_out = self.build_bottleneck_layer(inputs, growth_rate, is_training, keep_prob)

                out_put = self.build_composite_layer(bottleneck_out, growth_rate, kernel_h, kernel_w, slide_h, slide_w, is_training, keep_prob, name = "bn_relu_conv")

            else:

                out_put = self.build_composite_layer(inputs, growth_rate, kernel_h, kernel_w, slide_h, slide_w, is_training, keep_prob, name = "bn_relu_conv")

            output = tf.concat(axis=3, values=(inputs, out_put))

            return output


    #构造 bn + relu + conv 的混合 layer
    def build_composite_layer(self, inputs, output_channel_number, kernel_h, kernel_w, slide_h, slide_w, is_training, keep_prob, name = "bn_relu_conv"):

        with tf.variable_scope(name):

            outputs = getLayer.batch_normalization_layer(inputs, is_training, "bn")

            outputs = getLayer.relu_layer(outputs, "relu")

            outputs = getLayer.convolution_layer(outputs, output_channel_number, kernel_h, kernel_w, slide_h, slide_w, "conv")

            outputs = getLayer.dropout_layer(outputs, keep_prob, is_training)

            return outputs

    #构建 bottle neck layer
    def build_bottleneck_layer(self, inputs, growth_rate, is_training, keep_prob, name="bottleneck"):

        with tf.variable_scope(name):

            output_channel_number = growth_rate * 4

            outputs = getLayer.batch_normalization_layer(inputs, is_training, "bn")

            outputs = getLayer.relu_layer(outputs, "relu")

            outputs = getLayer.convolution_layer(outputs, output_channel_number, 1, 1, 1, 1, "conv")

            outputs = getLayer.dropout_layer(outputs, keep_prob, is_training)

            return outputs