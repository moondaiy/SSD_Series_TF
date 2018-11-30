#coding:utf-8
import tensorflow as tf
from .parameterDefaultConfig import *



def l2_regularizer(weight_decay=DEFAULT_WEIGHT_DECAY, scope=None):

    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                             dtype=tensor.dtype.base_dtype,
                                             name='weight_decay')

            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

    return regularizer


def l1_regularizer(weight_decay=DEFAULT_WEIGHT_DECAY, scope=None):

    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l1_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                             dtype=tensor.dtype.base_dtype,
                                             name='weight_decay')

            return tf.multiply(l2_weight, tf.reduce_sum(tf.abs(tensor)), name='value')

    return regularizer


def get_regularizer(regular=DEFAULT_REGULAR, weight_decay=DEFAULT_WEIGHT_DECAY, scope=None):

    regularizer = None

    if regular == "L2":

        regularizer = l2_regularizer(weight_decay=weight_decay, scope=None)

    elif regular == "L1":

        regularizer = l1_regularizer(weight_decay=weight_decay, scope=None)

    elif regular == "None":

        regularizer = None
    else:

        assert False, "No regular"

    return  regularizer


def get_variable(name, shape, initializer=None, trainable=True, regularizer=None):

    var = tf.get_variable(name, shape, initializer=initializer, trainable = trainable, regularizer=regularizer)

    return var

def get_variable_with_fixed_value(name, value, trainable=True , regularizer=None):

    var = tf.get_variable(name, initializer=value, trainable=trainable, regularizer=regularizer)

    return var

def get_convolution_variable(channel_in, channel_out, kernel_h, kernel_w, biased=True, trainable=True, regular=DEFAULT_REGULAR, weight_decay=DEFAULT_WEIGHT_DECAY, init_mode=DEFAULT_INITMODE):

    #以后改进选择不同的初始化方式进行不同的初值设置
    # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)

    if init_mode == "X":
        init_weights = tf.variance_scaling_initializer(scale=1.0, mode="fan_in",distribution="normal", seed=None, dtype=tf.float32)
    elif init_mode == "N":
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
    else:
        assert False, "Invalid Init Parameter"


    init_biases = tf.constant_initializer(0.0)

    kernel = get_variable('weights', [kernel_h, kernel_w, channel_in, channel_out], \
                          init_weights, trainable, regularizer=get_regularizer(regular, weight_decay))

    if biased == True:

        biases = get_variable('biases', [channel_out], init_biases, trainable)

    else:

        biases = None


    return kernel, biases



def get_full_connection_variable(sizeIn, sizeOut,biased=True,trainable=True, regular=DEFAULT_REGULAR, weight_decay = DEFAULT_WEIGHT_DECAY , initMode=DEFAULT_INITMODE):

    # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
    init_weights = tf.variance_scaling_initializer(scale=1.0, mode="fan_in", distribution="normal", seed=None,dtype=tf.float32)
    init_biases = tf.constant_initializer(0.0)

    weights = get_variable('weights', [sizeIn, sizeOut], init_weights, trainable,regularizer=get_regularizer(regular, weight_decay))

    if biased == True:

        biases = get_variable('biases', [sizeOut], init_biases, trainable)

    else:

        biases = None

    return weights, biases

def get_batch_normalization_variable(shape):

    beta = get_variable('beta', shape, initializer=tf.zeros_initializer)

    gamma = get_variable('gamma', shape, initializer=tf.ones_initializer)

    moving_mean = get_variable('moving_mean', shape, initializer=tf.zeros_initializer, trainable=False)

    moving_variance = get_variable('moving_variance', shape, initializer=tf.ones_initializer, trainable=False)

    return beta, gamma, moving_mean, moving_variance

def get_group_normalization_variable(shape):

    beta = get_variable('beta', shape, initializer=tf.zeros_initializer)

    gamma = get_variable('gamma', shape, initializer=tf.ones_initializer)

    return beta, gamma

def get_l2_normalization_variable(shape,scale_factor = 1.0):

    # init_value = tf.constant(scale_factor, shape=shape)
    #
    # scale = get_variable_with_fixed_value('scale', init_value)
    scale = get_variable('scale', shape, initializer=tf.ones_initializer())

    return scale * scale_factor