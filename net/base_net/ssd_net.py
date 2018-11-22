# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from . import ssd_vgg as ssd_vgg_net
from net.box_utils.boxes_tf_op import decode_boxes
from net.box_utils.nms_tf_op import nms_calculate


class SSD_Net(object):

    def __init__(self, base_net_info, anchor_info, extract_feature_info, all_anchor = None, loss_info = None, inference_flag = False, graph_create_flag = False):
        """
        :param inputs:
        :param is_training: 是否是训练状态
        :param base_net_type:
        :param size:
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if graph_create_flag == True:

            self.graph = tf.Graph()

        else:
            self.graph = tf.get_default_graph()

        with self.graph.as_default():

            self.base_net_info = base_net_info
            self.loss_info = loss_info

            self.base_net_type = base_net_info["base_net_type"]
            self.base_net_size = base_net_info["base_net_size"]
            self.class_number  = base_net_info["class_number"]

            self.train_flag = base_net_info["train_step"]
            self.anchors_info = anchor_info

            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.base_net_size, self.base_net_size, 3])

            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

            self.ssd_instance = self.build_ssd_instance(self.base_net_type, self.base_net_size)

            #创建基础特征提取网络
            self.base_net_out = self.build_base_feature_layer(self.base_net_type)
            self.valid_feature_layer_info = extract_feature_info


            #在基础网络的基础上进一步提取有效的feature layer
            self.valid_feature_out = self.build_valid_feature_layer(self.base_net_out, self.valid_feature_layer_info["extract_feature_valid_layer"], self.base_net_size)

            #外部传入anchors
            self.anchors = np.concatenate(all_anchor, axis=0)

            self.total_anchor_number = len(self.anchors)

            self.labels = tf.placeholder(dtype=tf.float32, shape=[ None, self.total_anchor_number, self.class_number + 9])

            #再进行最终输出的时候 做nms需要进行的门限值 ,也是在测试或inference阶段需要
            self.select_threshold = tf.placeholder(dtype=tf.float32, shape=[])
            self.nms_threshold = tf.placeholder(dtype=tf.float32, shape=[])

            #[0] 保存 loc
            #[1] 保存 cls
            # multibox_layer_out 给 计算loss的时候使用
            self.multibox_layer_out = self.build_multibox_layer(self.valid_feature_out, self.valid_feature_layer_info["extract_feature_valid_layer"], self.class_number, self.anchors_info["anchor_size"], self.anchors_info["anchor_ratios"], self.valid_feature_layer_info["extract_feature_normalization"], self.anchors_info["feature_shape"])

            self.net_out = self.build_net_out(self.multibox_layer_out)

            self.finally_box = self.build_visual_layer(self.net_out, self.anchors.astype(np.float32), self.anchors_info["prior_scaling"], self.select_threshold, self.nms_threshold)


        self.session = tf.Session(graph=self.graph, config=config)  # connect


    def build_ssd_instance(self, base_net_type, base_net_size):
        #主要是区分基础网络结构部分 vgg  ResNet等
        if base_net_type == "vgg":

            net = ssd_vgg_net.SSD_VGG(base_net_size)

        else:

            raise Exception("Only support vgg as base net type !")

        return net

    def build_valid_feature_layer(self, base_net_feature, valid_feature_list , base_net_size):

        valid_feature_layer = self.ssd_instance.build_valid_feature_layer(base_net_feature, valid_feature_list, base_net_size)

        return valid_feature_layer

    def build_anchors(self, anchor_infos, image_size):

        anchors = self.ssd_instance.build_anchors_with_anchor_infos(anchor_infos, image_size)

        return anchors


    def build_base_feature_layer(self, base_net_type):

        feature_layer_out = self.ssd_instance.build_feature_layer(self.inputs, self.is_training, self.base_net_size)

        return feature_layer_out

    def build_multibox_layer(self, input_list, input_name_list, class_number, anchor_base_size_list, anchor_ratio_list, normalization_list, feature_size_list):

        multibox_layer_out = self.ssd_instance.build_multibox_layer(input_list, input_name_list, class_number, anchor_base_size_list, anchor_ratio_list, normalization_list, feature_size_list)

        return multibox_layer_out


    def get_base_net_out(self):

        return self.base_net_out

    def build_loss(self, logits, labels, total_anchor_number, neg_pos_ratio = 3.0, min_negative_number = 0, alpha = 1.0):

        total_localization_loss, total_classification_loss, total_loss = self.ssd_instance.build_ssd_loss(logits, labels, total_anchor_number, neg_pos_ratio, min_negative_number, alpha)

        return total_localization_loss, total_classification_loss, total_loss

    def build_net_out(self, logistic_tensor):

        predict_tensor = self.ssd_instance.build_ssd_net_out(logistic_tensor)

        return predict_tensor


    #T.B.D ......
    def build_visual_layer(self, net_out_tesnor, anchor_tensor, scale_factors, select_threshold, nms_threshold):

        #初始化输出数据
        #这个是整个SSD的输出 label 0 - 20 数字 采用tensor array进行保存
        label_out = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)

        #box是具体的box坐标, 采用tensor array进行保存
        box_out = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        #置信度得分计算, 采用tensor array进行保存
        score_box = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        select_index = tf.TensorArray(dtype=tf.bool, size=1, dynamic_size=True)


        #有多少个 anchor
        total_anchor_number = tf.shape(anchor_tensor)[0]


        def body(i , label_out, box_out, score_box, select_index, net_out_tesnor):
            """
            :param i:  当前迭代次数,也就是在一个batch size中第几张图片
            :param label_out:  输出的label
            :param box_out:    输出的box_out
            :param score_box:  输出的 score_box
            :param net_out_tesnor:
            :return:
            """

            current_net_out = net_out_tesnor[i]

            current_predict_object_label = current_net_out[:,1:21]
            current_predict_encode_bbox  = current_net_out[:,21:25]

            #解码操作
            current_predict_decode_bbox  = decode_boxes(current_predict_encode_bbox, anchor_tensor, scale_factors)


            #计算分类置信度最大的分类 得分
            current_predict_max_object_scores = tf.reduce_max(current_predict_object_label, axis=1)

            current_predict_max_object_label  = tf.argmax(current_predict_object_label, axis=1, output_type=tf.int32) + 1

            #如果 object 置信度 > select_threshold 则保留该项目 否则 设置成 0 object的置信度 > select_threshold 其他的为 0.1
            valid_object_scores = tf.where(current_predict_max_object_scores > select_threshold, current_predict_max_object_scores, tf.ones_like(current_predict_max_object_scores) * 0.1)

            #保存当前image中的 scores
            score_box = score_box.write(i, valid_object_scores)


            valid_object_label = tf.where(current_predict_max_object_scores > select_threshold, current_predict_max_object_label, tf.zeros_like(current_predict_max_object_label, dtype=tf.int32))
            label_out = label_out.write(i, valid_object_label)

            #那些无效的box 的相对位置信息 -> [0,0,1,1] scores = 0.1 因此在最终显示的时候 要注意处理下 或者查看输出的label分数
            #有效的box 信息 ymin xmin ymax xmax 的形式 ...
            valid_object_bbox  = tf.where(current_predict_max_object_scores > select_threshold, current_predict_decode_bbox, tf.tile(tf.constant([[0., 0. , 1., 1.]], dtype=tf.float32), multiples=[total_anchor_number, 1]))
            box_out = box_out.write(i, valid_object_bbox)

            valid_select_index = nms_calculate(valid_object_bbox, valid_object_scores, nms_threshold, total_anchor_number)
            valid_select_index = tf.sparse_to_dense(valid_select_index, [total_anchor_number], True, False, validate_indices=False)
            select_index = select_index.write(i, valid_select_index)


            i = i + 1

            return i , label_out, box_out, score_box, select_index, net_out_tesnor

        def cond(i , label_out, box_out, score_box, select_index, net_out_tesnor):

            r = tf.less(i, tf.shape(net_out_tesnor)[0])

            return r

        i = 0

        i, label_out, box_out, score_box, select_index, net_out_tesnor = tf.while_loop(cond, body, [i , label_out, box_out, score_box, select_index, net_out_tesnor])

        label_out = label_out.stack()
        box_out   = box_out.stack()
        score_box = score_box.stack()
        select_index = select_index.stack()


        return label_out, box_out, score_box, select_index


    def get_graph(self):

        return self.graph


    def get_session(self):

        return self.session









