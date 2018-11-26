# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os

from . import image_preprocess
from net.box_utils import train_bbox_anchor_tf_op
import glob




class Data_Manager(object):

    def __init__(self, data_record_path, batch_size , is_training, image_size, anchors, class_numer, scale_factors, anchor_pos_iou, data_format = "NHWC"):

        self.data_record_path = data_record_path
        self.batch_size = batch_size
        self.is_training = is_training
        self.image_size = image_size
        self.anchor_number = len(np.concatenate(anchors, axis=0))
        self.anchors_tensor = tf.constant(np.concatenate(anchors, axis=0),dtype=tf.float32)
        self.class_number = class_numer
        self.anchor_scale = scale_factors
        self.anchor_pos_iou = anchor_pos_iou
        self.total_sample_number = self.countTFRecordSampleNumber(self.data_record_path)

        self.img_name_batch, self.img_batch, self.gtboxes_and_label_batch_float, self.num_obs_batch, self.img_height, self.img_width = \
            self.init_data_manager(self.batch_size, self.image_size, self.is_training, self.anchors_tensor, data_format)


    def read_single_example_and_decode(self, filename_queue):

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized=serialized_example,
            features={
                'img_name': tf.FixedLenFeature([], tf.string),
                'img_height': tf.FixedLenFeature([], tf.int64),
                'img_width': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
                'num_objects': tf.FixedLenFeature([], tf.int64)
            }
        )

        img_name = features['img_name']
        img_height = tf.cast(features['img_height'], tf.int32)
        img_width = tf.cast(features['img_width'], tf.int32)
        img = tf.decode_raw(features['img'], tf.uint8)

        img = tf.reshape(img, shape=[img_height, img_width, 3])

        gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
        # gtboxes_and_label.set_shape([None, 5])

        num_objects = tf.cast(features['num_objects'], tf.int32)

        return img_name, img, gtboxes_and_label, num_objects , img_height, img_width

    #在训练阶段,去预处理训练用到的图片
    def process_image_for_training(self, filename_queue, image_size, data_format):


        #读一张图片的信息
        img_name, img, gtboxes_and_label, num_objects, img_height, img_width = self.read_single_example_and_decode(filename_queue)

        #box坐标信息 转换成 相对于图像 宽度和高度的信息
        #gt_box_and_label_tensor ymin xmin ymax xmax 相对原始宽度和高度
        gt_box_and_label_tensor = image_preprocess.box_info_normilization(gtboxes_and_label, img_height, img_width)

        #随机的采样
        #经过随机采样后的形成的box
        image_tensor, labels, bboxes, distort_bbox =\
            image_preprocess.sample_distorted_bounding_box_crop(img, tf.reshape(gt_box_and_label_tensor[:,4], shape=[-1,1]), gt_box_and_label_tensor[:,:4] ,min_object_covered=0.05,aspect_ratio_range=(1.0, 1.0))

        #随机左右翻转操作
        #此时box 坐标会变化
        image_tensor, bboxes = image_preprocess.random_flip_left_right(image_tensor, bboxes)

        #在最后一个步骤进行resize操作
        image_tensor = image_preprocess.resize_image_with_fixed_size(image_tensor, image_size, image_size,
                                                                          method=tf.image.ResizeMethod.BILINEAR,
                                                                          align_corners=False)

        image_tensor = image_preprocess.convert_image_format(image_tensor)

        # #随机进行对图片的修正 ,像素变更为0 到 1之间的类型
        image_tensor = image_preprocess.random_adjust_image_pixes(image_tensor)

        image_tensor = image_preprocess.convert_image_format(image_tensor,type="0_255")

        #去中心化操作
        image_tensor = image_preprocess.image_whitened(image_tensor)

        image_tensor = image_preprocess.random_adjust_image_pixes(image_tensor)

        # img = img - tf.constant([103.939, 116.779, 123.68])
        #
        #对于训练阶段和实际使用阶段的话. 图像预处理策略是不同的
        # if is_training:
        #
        #     img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img, gtboxes_and_label=gtboxes_and_label)
        #
        # else:
        #
        #     img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
        #                                                                 target_shortside_len=image_size)

        #是否转换
        if data_format == "NCHW":

            image_tensor = tf.transpose(image_tensor, (2,0,1))


        #合并label和gt box信息
        gt_box_and_label_tensor = tf.concat([bboxes, labels],axis=1)

        return img_name, image_tensor, gt_box_and_label_tensor, num_objects , img_height, img_width


    def process_image_for_testing(self,filename_queue, image_size, data_format):
        pass

    def read_and_prepocess_single_img(self, filename_queue, image_size, is_training=True, data_format ="NHWC"):

        if is_training == True:

            img_name, distort_image_tensor, gt_box_and_label_tensor, num_objects, img_height, img_width = \
                self.process_image_for_training(filename_queue, image_size, data_format)

        else:

            img_name, distort_image_tensor, gt_box_and_label_tensor, num_objects, img_height, img_width = \
                self.process_image_for_training(filename_queue, image_size, data_format)



        return img_name, distort_image_tensor, gt_box_and_label_tensor, num_objects, img_height, img_width



    def encode_ground_truth_box_label_for_train(self, anchors_tensor, anchor_number, gtboxes_and_label_tensor, scale_factors , class_number , pos_iou_threshold = 0.5):

        return train_bbox_anchor_tf_op.generate_train_encoding_labels_tf_operation(anchors_tensor, anchor_number, gtboxes_and_label_tensor[:,:4], gtboxes_and_label_tensor[:,4], scale_factors, class_number)

    def init_data_manager(self, batch_size, image_size, is_training, anchors, data_format):

        #这种写法只能是训练过程中不能进行验证或者test操作.
        pattern = os.path.join(self.data_record_path, '*.tfrecord')

        print('tfrecord path is -->', os.path.abspath(pattern))

        filename_tensorlist = tf.train.match_filenames_once(pattern)

        filename_queue = tf.train.string_input_producer(filename_tensorlist)

        img_name, img, gtboxes_and_label_float, num_obs, img_height, img_width = self.read_and_prepocess_single_img(filename_queue, image_size, is_training=is_training, data_format= data_format)

        #将label信息进行编码操作
        encode_gt_box_label_tensor = self.encode_ground_truth_box_label_for_train(self.anchors_tensor, self.anchor_number, gtboxes_and_label_float, self.anchor_scale, self.class_number, self.anchor_pos_iou)

        img_name_batch, img_batch, encode_gt_box_label_tensor, num_obs_batch, img_height, img_width = \
            tf.train.batch(
                           [img_name, img, encode_gt_box_label_tensor, num_obs, img_height, img_width],
                           batch_size=batch_size,
                           capacity=50,
                           num_threads=4,
                           dynamic_pad=False)

        # img_tensor_with_bound_box_tensor = tf.image.draw_bounding_boxes(img_batch, gtboxes_and_label_batch_float[:,:4])

        return img_name_batch, img_batch, encode_gt_box_label_tensor, num_obs_batch, img_height, img_width


    def next_batch(self):

        return self.img_name_batch, self.img_batch, self.gtboxes_and_label_batch_float, self.num_obs_batch, self.img_height, self.img_width

    def countTFRecordSampleNumber(self, tfrecordPath):

        tf_record_path_pattern = tfrecordPath + "/*.tfrecord"

        recordList = glob.glob(tf_record_path_pattern)

        if recordList is None:

            raise Exception("Count The Number Of Sample Failure")

        number = 0

        for tfrecord in recordList:

            for record in tf.python_io.tf_record_iterator(tfrecord):
                number = number + 1

        return number

