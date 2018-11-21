# -*- coding: utf-8 -*-
import tensorflow as tf
from libs import getLayer as layer
from base_net import ssd_base_net as base_net

#定义SSD网络传递参数
#
class SSD(object):

    def __init__(self, base_net_type = "vgg", inference_flag = False, graph_create_flag = True, size = 300, refined_flag=False):
        #graph_create_flag : 是否独自创建 graph
        #size : 网络大小 300 或 512
        #inference_flag : 是否则ference阶段

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #如果是True 则单独创建一个graph
        if graph_create_flag == True:

            self.graph = tf.Graph()

        else:
            self.graph = tf.get_default_graph()


        with self.graph.as_default():
            #再graph下创建网络或者恢复参数等操作

            #数据
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

            #label
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])

            #是否处于训练状态
            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

            self.base_net_out = self.build_base_net(self.inputs, base_net_type)

            self.anchors = None


        self.session = tf.Session(graph=self.graph, config=config)  # 定义Session

        if inference_flag == True:
            pass



    def build_base_net(self, inputs, is_training, base_net_type = "vgg", size = 300):

        #type表示 基础网络类型

        return base_net.SSD_Base_Net(inputs, is_training, base_net_type, size).get_base_net_out()


    def build_loss(self, logits=None, labels=None):
        pass


    def train(self, train_args):
        pass


if __name__=="__main__":

    ssd_net = SSD()

