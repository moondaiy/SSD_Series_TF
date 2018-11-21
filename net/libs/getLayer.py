#coding:utf-8
import tensorflow as tf
from functools import reduce
from .getParameter import *
from .parameterDefaultConfig import *
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


#update bn layer parameters
LAYERS_UPDATE_OPS_COLLECTION ="LAYERS_EXTRA_UPDATE_OPS"


def check_padding_valid(padding):
    assert (padding in ALL_PADDINT)

def check_regular_valid(regular):
    assert (regular in ALL_REGULAR)

def check_rnn_style(style):
    assert (style in ALL_LSTM_STYLE)

def convolution_layer(input, channel_out, kernel_h, kernel_w, slide_h, slide_w, name, biased=True, activationFn = None, padding=DEFAULT_PADDING, trainable=True, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    #检查当前Layer的操作,是由当前层进行
    check_padding_valid(padding)
    check_regular_valid(regular)

    channel_in = input.get_shape()[-1]

    ret_value = None

    #这里加上了 tf.variable_scope(name) 的限制,则get_convolution_variable中就不需要加限制,以后也是这样
    with tf.variable_scope(name) as scope:

        kernel,biases = get_convolution_variable(channel_in, channel_out, kernel_h, kernel_w, biased=biased, trainable=trainable, regular=regular, weight_decay = weight_decay, init_mode=init_mode)

        conv = tf.nn.conv2d(input, kernel, [1, slide_h, slide_w, 1], padding=padding)

        if biased == True:

            conv_plus_bias = tf.nn.bias_add(conv, biases)

            if activationFn != None:

                ret_value =  activationFn(conv_plus_bias)

            else:
                ret_value =  conv_plus_bias
        else:

            if activationFn != None:

                conv =  activationFn(conv, name=scope.name)

            ret_value = conv

    return ret_value

def depthwise_convolution_layer(input, depth_wise_conv_kernel_h, depth_wise_conv_kernel_w, depth_wise_conv_multipler,slide_h, slide_w, name , biased=False, activationFn = None, padding=DEFAULT_PADDING, trainable=True, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    channel_in = input.get_shape()[-1]

    with tf.variable_scope(name):

        depthwise_filter, _ = get_convolution_variable(channel_in, depth_wise_conv_multipler, depth_wise_conv_kernel_h,
                                                       depth_wise_conv_kernel_w, biased=biased, trainable=trainable,
                                                       regular=regular, weight_decay=weight_decay, init_mode=init_mode)

        outs = tf.nn.depthwise_conv2d(input, depthwise_filter, [1,slide_h,slide_w,1], padding=padding)

        if activationFn != None:
            outs = activationFn(outs)

        return outs


def separable_convolution_layer(input, point_wise_conv_out, depth_wise_conv_kernel_h, depth_wise_conv_kernel_w, depth_wise_conv_multipler,slide_h, slide_w, name, is_train , alpha = 1, biased=False, activationFn = tf.nn.relu6, padding=DEFAULT_PADDING, trainable=True, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):



    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('alpha can be one of'
                         '`0.25`, `0.50`, `0.75` or `1.0` only.')

    channel_in = input.get_shape()[-1]

    #决定实际的输出channel
    if point_wise_conv_out != None:
        pointwise_filter_channel_in = alpha * point_wise_conv_out
    else:
        pointwise_filter_channel_in = 0

    #加入域名
    with tf.variable_scope(name) as scope:

        with tf.variable_scope("depth_wise_conv"):

            #channel multipler is 1
            depthwise_filter, _ = get_convolution_variable(channel_in, depth_wise_conv_multipler, depth_wise_conv_kernel_h, depth_wise_conv_kernel_w, biased=biased, trainable=trainable, regular=regular, weight_decay = weight_decay, init_mode=init_mode)

            net = tf.nn.depthwise_conv2d(input, depthwise_filter, [1, slide_h, slide_w, 1], padding=padding)

            net = batch_normalization_layer(net, is_train, "bn")


            if activationFn != None:

                net = activationFn(net)

        # 如果 channel_out 不存在 则不执行 point_wise_conv 过程
        if point_wise_conv_out != None:
            with tf.variable_scope("point_wise_conv"):

                pointwise_filter, _ = get_convolution_variable(pointwise_filter_channel_in, point_wise_conv_out, 1, 1, biased=False, trainable=trainable, regular=regular, weight_decay = weight_decay, init_mode=init_mode)

                net = tf.nn.conv2d(net, pointwise_filter, [1, slide_h, slide_w, 1], padding=padding)

                net = batch_normalization_layer(net, is_train, "bn")

                if activationFn != None:
                    net = tf.nn.relu(net)

        return net


def full_connection_Layer(input, out_size, name, biased=True, activationFn=tf.nn.relu, trainable=True, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    #获取输入维度,如果input是卷积的输出,则进行全连接的操作的时候,需要拉平整个tensor,如果input本身就是全连接的输出,则也是拉平,但是因为维度不一致所以需要连乘操作
    #这地方需要修正
    size_in = reduce(lambda x,y: x*y, input.get_shape()[1:]) #少个连乘操作
    ret_value = None

    with tf.variable_scope(name) as scope:

        weights, biases = get_full_connection_variable(size_in, out_size, biased, trainable, regular, weight_decay, init_mode)

        input_mul_w = tf.matmul(input, weights)

        if biased == True:

            fc = tf.nn.bias_add(input_mul_w, biases)

        else:

            fc = input_mul_w

        if activationFn != None:

            ret_value = activationFn(fc)
        else:
            ret_value = fc

        return ret_value

#获取单个rnn单元
def get_rnn_cell(sizeHiden, cellStyle=DEFAULT_RNN_CELL_STYLE, rnn_keep_prob=DEFAULT_RNN_KEEP_PROB, num_proj=None):

    if cellStyle=="LSTM":

        cell = tf.nn.rnn_cell.LSTMCell(num_units=sizeHiden, num_proj=num_proj)

    elif cellStyle=="GRU":

        cell = tf.nn.rnn_cell.GRUCell(num_units=sizeHiden)

    else:
        assert False, "RNN Cell Style Error"

    # 如果需要rnn drop out 则进行drop out操作
    if rnn_keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=rnn_keep_prob)

    return cell


#将CNN提取到的特征数据转换成LSTM可用的数据形式,并得到 seq_len
def cnn_map_lstm_layer(input):

    input_shape = tf.shape(input)

    batch_size = input_shape[0]
    input_h    = input_shape[1]
    input_w    = input_shape[2]
    input_c    = input_shape[3]

    seq_len = tf.ones([batch_size], tf.int32) * input_w

    inputTransposed = tf.transpose(input, [0, 2, 1, 3])

    input_transposed_reshape = tf.reshape(inputTransposed, [batch_size, input_w, input_h * input_c])

    cnn_shape = input.get_shape().as_list()

    input_transposed_reshape.set_shape([None, cnn_shape[2], cnn_shape[1] * cnn_shape[3]])

    return input_transposed_reshape, seq_len

#此时input的格式 需要变换为[batchsize, maxtimestep, feature]
def lstm_layer(input, hiden_size, out_size, name, seq_length, rnn_style= DEFAULT_RNN_CELL_STYLE, rnn_keep_prob = DEFAULT_RNN_KEEP_PROB, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    #检查RNN类型是否支持
    check_rnn_style(rnn_style)

    with tf.variable_scope(name) as scope:

        #解析输入tensor的维度信息
        shape = tf.shape(input)

        batch_size , time_step_size , feature_size = shape[0], shape[1], shape[2]

        #根据输入参数获得rnn基本单元,目前可选择单元为GRU或者LSTM
        lstm_cell = get_rnn_cell(hiden_size, cellStyle=rnn_style, rnn_keep_prob=rnn_keep_prob)

        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, input, initial_state=initial_state, dtype=tf.float32,sequence_length=seq_length)

        lstm_out = tf.reshape(lstm_out, [-1, hiden_size])

        #lstm内部的全连接层默认是可训练且带有 biased
        weights, biases = get_full_connection_variable(hiden_size, out_size, True, True, regular, weight_decay, init_mode)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [shape[0], -1, out_size])

        return outputs



def bi_lstm_layer(input, hiden_size, out_size, name, seq_length, rnnStyle= DEFAULT_RNN_CELL_STYLE, rnn_keep_prob = DEFAULT_RNN_KEEP_PROB, \
                  regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    check_rnn_style(rnnStyle)

    with tf.variable_scope(name) as scope:

        shape = tf.shape(input)

        lstm_fw_cell = get_rnn_cell(hiden_size, cellStyle=rnnStyle, rnn_keep_prob=rnn_keep_prob)
        lstm_bw_cell = get_rnn_cell(hiden_size, cellStyle=rnnStyle, rnn_keep_prob=rnn_keep_prob)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32, sequence_length=seq_length)

        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [-1, 2 * hiden_size])

        #进行一次全连接维度映射到输出维度
        weights, biases = get_full_connection_variable(2 * hiden_size, out_size, True, True, regular, weight_decay, init_mode)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [shape[0], -1, out_size])

        return outputs


def batch_normalization_layer(input, is_train, name, is_convolution=True, moving_decay = DEFAULT_BN_MOVING_DECAY, bn_decay= DEFAULT_BN_EPSILON):

    shape = input.get_shape().as_list()[-1]  # 根据channel得到

    with tf.variable_scope(name) as scope:

        # 针对不同的类型,得到的 mean 和 variance 是不同的
        if is_convolution:  # 针对卷积的BN 简单理解 有多少个卷积核 就有多少个mean 和 variance
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
        else:
            mean, variance = tf.nn.moments(input, axes=[0])

        beta,gamma, moving_mean,moving_variance = get_batch_normalization_variable(shape)

        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, moving_decay)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, moving_decay)

        tf.add_to_collection(LAYERS_UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(LAYERS_UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(is_train, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

        output = tf.nn.batch_normalization(input, mean, variance, beta, gamma, bn_decay)

        return output


def relu_layer(input, name = "relu"):

    return tf.nn.relu(input, name)

# def dropOutLayer(input, keepProb):
#
#     return tf.nn.dropout(input, keepProb)

def dropout_layer(input, keepProb, is_training):

    if keepProb < 1:

        output = tf.cond(is_training,lambda: tf.nn.dropout(input, keepProb),lambda: input)

    else:

        output = input

    return output


def average_pooling_layer(input, kernel_h = 2, kernel_w = 2, slide_h = 2, slide_w = 2, padding=DEFAULT_PADDING):

    check_padding_valid(padding)

    return tf.nn.avg_pool(input, [1, kernel_h, kernel_w, 1], [1, slide_h, slide_w, 1], padding)


def max_pooling_layer(input, kernel_h = 2, kernel_w = 2, slide_h = 2, slide_w = 2, padding=DEFAULT_PADDING):

    check_padding_valid(padding)

    return tf.nn.max_pool(input, [1, kernel_h, kernel_w, 1], [1, slide_h, slide_w, 1], padding)


def deconv2d_layer(input, channel_out, kernel_h, kernel_w, slide_h, slide_w, name,
                   biased=True, activationFn = None, padding=DEFAULT_PADDING, trainable=True,
                   regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    check_padding_valid(padding)
    check_regular_valid(regular)

    shape = input.get_shape().as_list()

    channel_in = shape[3]

    batch_size = tf.shape(input)[0]
    height = tf.shape(input)[1] * slide_h
    width = tf.shape(input)[2] * slide_w
    shape_out = tf.stack([batch_size, height, width, channel_out])


    with tf.variable_scope(name):

        kernel,biases = get_convolution_variable(channel_out, channel_in, kernel_h, kernel_w, biased=biased, trainable=trainable, regular=regular, weight_decay=weight_decay, init_mode=init_mode)

        deconv = tf.nn.conv2d_transpose(input, kernel, shape_out,  [1, slide_h, slide_w, 1], padding=padding)

        if biased is True:

            deconv = tf.nn.bias_add(deconv, biases)

        if activationFn != None:

            deconv = activationFn(deconv)


        return deconv


def atrous_conv_2d_layer(input, channel_out, kernel_h, kernel_w, rate, name,
                   biased=True, activationFn = None, padding=DEFAULT_PADDING, trainable=True,
                   regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    #检查当前Layer的操作,是由当前层进行
    check_padding_valid(padding)
    check_regular_valid(regular)

    channel_in = input.get_shape()[-1]

    ret_value = None

    with tf.variable_scope(name) as scope:

        kernel,biases = get_convolution_variable(channel_in, channel_out, kernel_h, kernel_w, biased=biased, trainable=trainable, regular=regular, weight_decay = weight_decay, init_mode=init_mode)

        conv = tf.nn.atrous_conv2d(input, kernel, rate, padding=padding)

        if biased == True:

            conv_plus_bias = tf.nn.bias_add(conv, biases)

            if activationFn != None:

                ret_value =  activationFn(conv_plus_bias)

            else:
                ret_value =  conv_plus_bias
        else:

            if activationFn != None:

                conv =  activationFn(conv, name=scope.name)

            ret_value = conv

    return ret_value


def pad_2d(net, pad = (0,0), name_scope = "pad", mode='CONSTANT',data_format='NHWC'):

    with tf.variable_scope(name_scope) as scope:

        if data_format == 'NHWC':

            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]

        elif data_format == 'NCHW':

            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]

        net = tf.pad(net, paddings, mode=mode)

        return net


def normalization_l2_layer(inputs, scale_factor, scope_name):

    with tf.variable_scope(scope_name) as scope:

        inputs_shape = inputs.get_shape()

        inputs_rank = inputs_shape.ndims

        norm_dim = tf.range(inputs_rank - 1, inputs_rank)

        params_shape = inputs_shape[-1:]

        outputs = tf.nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)

        if scale_factor > 0:

            scale_factor = get_l2_normalization_variable(params_shape, scale_factor)

        return outputs * scale_factor




