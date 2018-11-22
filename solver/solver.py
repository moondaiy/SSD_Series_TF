# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from net.data_manager.data_manager import Data_Manager
from tools.model_restore import do_restore_ckpt
from tools.model_restore import do_meta_file_exist
import math
from net.libs.getLayer import LAYERS_UPDATE_OPS_COLLECTION

#对外提供一个 train方法进行所有的训练操作

class Solver(object):

    def __init__(self, net:"SSD_Net" , data_manager:"Data_Manager", anchors, config_info:"tuple"):

        self.config_info   = config_info
        self.base_info     = self.config_info[0]
        self.anchor_info   = self.config_info[1]
        self.loss_info     = self.config_info[3]
        self.training_info = self.config_info[4]



        self.model         = net
        self.data_provider = data_manager

        self.train_max_epoch        = self.training_info["max_epoch"]
        self.train_batch_size       = self.training_info["batch_size"]
        self.train_init_learning    = self.training_info["learn_ratio"]
        self.train_batch_per_epoch  = int(self.data_provider.total_sample_number / self.train_batch_size)
        self.optimizer_type         = self.training_info["optimizer_type"]

        self.epoch_start_index = 0
        self.batch_start_index = 0
        self.restore_checkpoint_flag = self.training_info["train_restore_flag"]
        self.check_point_dir =  self.training_info["check_points"]
        self.use_batch_norm  =  self.training_info["use_batch_norm"]

        self.graph   = self.model.get_graph()
        self.session = self.model.get_session()

        if self.restore_checkpoint_flag == True:
            pass

        with self.graph.as_default():

            #训练用的 step 计数
            self.global_step = tf.Variable(0, trainable=False)

            #学习率配置
            self.lr = tf.constant(self.train_init_learning, dtype=tf.float32)

            #模型保存恢复和保存配置
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)


    def train(self):

        with self.graph.as_default():

            total_localization_loss, total_classification_loss, total_loss = self.model.build_loss(self.model.multibox_layer_out,self.model.labels,self.model.total_anchor_number)

            with tf.variable_scope("optimizer_vars"):

                if self.use_batch_norm == True:

                    with tf.control_dependencies(tf.get_collection(LAYERS_UPDATE_OPS_COLLECTION)):

                        optimizer = self.get_optimizer()

                        train_op = optimizer(learning_rate=self.train_init_learning).minimize(total_localization_loss, global_step=self.global_step)
                else:

                    with tf.control_dependencies(tf.get_collection(LAYERS_UPDATE_OPS_COLLECTION)):

                        optimizer = self.get_optimizer()

                        train_op = optimizer(learning_rate=self.train_init_learning).minimize(total_localization_loss, global_step=self.global_step)

            with self.session.as_default() as sess:

                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                if self.restore_checkpoint_flag == True:

                    self.restore_model()

                coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
                threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner，此时文件名队列已经进队

                last_loss = 100

                for epoch in range(self.epoch_start_index, self.train_max_epoch):

                    for batch in range(self.batch_start_index, self.train_batch_per_epoch):

                        #得到当前的 batch 数据
                        image_name_batch, image_batch, gt_label_batch, num_object, img_height, img_width = sess.run(self.data_provider.next_batch())

                        _, r_total_localization_loss, r_total_classification_loss, r_total_loss, global_step, learning_ratio = \
                            sess.run([train_op, total_localization_loss, total_classification_loss, total_loss, self.global_step, self.lr],feed_dict={self.model.labels: gt_label_batch, self.model.inputs: image_batch, self.model.is_training: True})

                        print("Current epoch %d train total step %d learn rate is %f total loss is %f classification loss is %f" %
                              (epoch, global_step, learning_ratio, r_total_loss, r_total_classification_loss))

                    self.batch_start_index = 0

                coord.request_stop()
                coord.join(threads)

    def restore_model(self):

        do_restore_ckpt(self.session, self.saver, self.check_point_dir)

        step_restored = self.session.run(self.global_step)

        # globalStep 是按照batch来计算的
        self.epoch_start_index = math.floor(step_restored / self.train_batch_per_epoch)

        self.batch_start_index = step_restored % self.train_batch_per_epoch

        print("Restored global step: %d" % step_restored)
        print("Restored epoch: %d" % self.epoch_start_index)
        print("Restored batch_start_index: %d" % self.batch_start_index)


    def save_checkpoint(self, step ,loss=None):

        ckpt_name = "global_step_%d" % step

        if loss is not None:
            ckpt_name += '_loss_%.04f' % loss

        name = os.path.join(self.check_point_dir, ckpt_name)

        print("save checkpoint %s" % name)

        meta_exists, meta_file_name = do_meta_file_exist(self.check_point_dir)

        self.saver.save(self.session, name)

        # remove old meta file to save disk space
        if meta_exists:
            try:
                os.remove(os.path.join(self.check_point_dir, meta_file_name))
            except:
                print('Remove meta file failed: %s' % meta_file_name)


    def get_optimizer(self):

        if self.optimizer_type == "adam_optimizer":

            optimizer = tf.train.AdamOptimizer
        else:
            assert False , "Now Only Support Adam Optimizer"

        return optimizer



