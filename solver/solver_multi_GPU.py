# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from data_manager.data_manager import  Data_Manager
from tools.model_restore import do_restore_ckpt
from tools.model_restore import do_meta_file_exist
import math
from net.libs.getLayer import LAYERS_UPDATE_OPS_COLLECTION

#对外提供一个 train方法进行所有的训练操作

class Solver_multiple_GPU(object):

    def __init__(self, anchors, config_info:"tuple"):

        self.config_info   = config_info
        self.base_info     = self.config_info[0]
        self.anchor_info   = self.config_info[1]
        self.extract_feature_info = self.config_info[2]
        self.loss_info     = self.config_info[3]
        self.training_info = self.config_info[4]

        self.gpu_number = self.training_info["gpu_number"]
        self.train_max_epoch        = self.training_info["max_epoch"]
        self.train_batch_size       = self.training_info["batch_size"] * self.gpu_number
        self.train_init_learning    = self.training_info["learn_ratio"]
        self.number_epoch_for_decay = self.training_info["number_epoch_for_decay"]
        self.decay_rate             = self.training_info["decay_rate"]

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        #在一个GPU中进行
        self.graph   = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph, config=config)

        #数据提供单元交给cpu进行
        with self.graph.as_default() , tf.device("/cpu:0"):
            self.data_provider = Data_Manager(self.training_info["tf_record_path"], self.training_info["batch_size"] * self.gpu_number, self.base_info["train_step"],
                                         self.base_info["base_net_size"], anchors.get_anchors(), self.base_info["class_number"],
                                         self.anchor_info["prior_scaling"], self.anchor_info["anchor_pos_iou_threshold"])

        image_name_batch, image_batch, gt_label_batch, num_object, img_height, img_width = self.data_provider.next_batch()

        image_batch_splits = tf.split(image_batch, self.gpu_number)
        gt_label_batch_splits = tf.split(gt_label_batch, self.gpu_number)

        self.model_list = []

        for i in range(self.gpu_number):

            with self.graph.as_default() , tf.device("/gpu:%d" % (i)), tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse= tf.AUTO_REUSE):

                model = SSD_Net(self.base_info, self.anchor_info, self.extract_feature_info, anchors.get_anchors(), self.loss_info,
                                extra_input=False , input_image_batch = image_batch_splits[i], input_label_batch=gt_label_batch_splits[i])

                # l2_loss_var = []
                #
                # for vars in tf.trainable_variables():
                #     if "beta" not in vars.name and "gamma" not in vars.name:
                #         a = tf.nn.l2_loss(vars)
                #         l2_loss_var.append(tf.nn.l2_loss(vars))

                self.model_list.append(model)



        self.train_batch_per_epoch  = int(self.data_provider.total_sample_number / self.train_batch_size)
        self.optimizer_type         = self.training_info["optimizer_type"]

        self.epoch_start_index = 0
        self.batch_start_index = 0
        self.restore_checkpoint_flag = self.training_info["train_restore_flag"]
        self.check_point_dir =  self.training_info["check_points"]
        self.use_batch_norm  =  self.training_info["use_batch_norm"]

        #sgd 优化
        self.learn_ratio_change_ratio = self.training_info["learn_ratio_change_ratio"]

        #更改学习率变更 steps
        self.learn_change_boundaries = self.training_info["learn_ratio_change_boundaries"]
        self.learn_change_boundaries = [value * self.learn_ratio_change_ratio for value in self.learn_change_boundaries]

        self.learn_ration_boundaries = []
        self.learn_ration_decay      = self.training_info["learn_ration_decay"]
        self.momentum                = self.training_info["momentum"]


        for i in range(len(self.learn_change_boundaries) + 1):

            self.learn_ration_boundaries.append(self.train_init_learning * pow(self.learn_ration_decay, i))

        with self.graph.as_default() , tf.device('/cpu:0'):

            #训练用的 step 计数
            self.global_step = tf.Variable(0, name="global_step" , trainable=False)

            # decay_steps = int(self.train_batch_per_epoch * self.number_epoch_for_decay)
            #07 + 12 22136 图片
            #学习率配置
            # self.lr = tf.train.exponential_decay(learning_rate = self.train_init_learning, global_step =  self.global_step,  decay_steps = decay_steps, decay_rate = self.decay_rate)
            self.lr = tf.train.piecewise_constant(self.global_step, boundaries=self.learn_change_boundaries, values=self.learn_ration_boundaries)

            #模型保存恢复和保存配置
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)



    def train(self):

        #m默认的计算设备是在cpu上

        with self.graph.as_default(), tf.device('/cpu:0'):

            with tf.variable_scope("optimizer_vars"):

                optimizer = self.get_optimizer()(self.lr, self.momentum)

            tower_grads = []
            total_localization_loss_list = []
            total_classification_loss_list = []
            total_loss_list = []

            for i in range(self.gpu_number):

                with tf.device('/gpu:%d' % i):
                    # with tf.name_scope('GPU_%d' % i) as scope:
                        with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

                            model = self.model_list[i]

                            total_localization_loss, total_classification_loss, total_loss = model.build_loss(model.multibox_layer_out,model.labels,model.total_anchor_number)

                            l2_loss_var = []

                            for vars in tf.trainable_variables():
                                if "beta" not in vars.name and "gamma" not in vars.name :#and "l2_scale" not in vars.name:

                                    if "l2_scale" not in vars.name:

                                        l2_loss_var.append(tf.nn.l2_loss(vars))

                                    else:

                                        l2_loss_var.append(tf.nn.l2_loss(vars) * 0.1)

                            #计算正则化损失函数
                            regular_loss = tf.add_n(l2_loss_var) * 0.0005

                            # regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            # regular_loss = tf.add_n(l2_loss_var)
                            total_loss = total_loss + regular_loss
                            # tf.get_variable_scope().reuse_variables()

                            grads = optimizer.compute_gradients(total_loss)

                            tower_grads.append(grads)

                            total_localization_loss_list.append(total_localization_loss)
                            total_classification_loss_list.append(total_classification_loss)
                            total_loss_list.append(total_loss)

            grads = self.average_gradients(tower_grads)

            total_localization_loss = self.average_loss(total_localization_loss_list)
            total_classification_loss = self.average_loss(total_classification_loss_list)
            total_loss = self.average_loss(total_loss_list)

            train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

            with self.session.as_default() as sess:

                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                if self.restore_checkpoint_flag == True:

                    self.restore_model()

                coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
                threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner，此时文件名队列已经进队

                last_loss = 3.5

                for epoch in range(self.epoch_start_index, self.train_max_epoch):

                    for batch in range(self.batch_start_index, self.train_batch_per_epoch):

                        _, r_total_localization_loss, r_total_classification_loss, r_total_loss, _global_step, learning_ratio = \
                            sess.run([train_op, total_localization_loss, total_classification_loss, total_loss, self.global_step, self.lr])

                        print("Current epoch %d train total step %d learn rate is %f total loss is %f classification loss is %f  localization loss is %f" %
                              (epoch, _global_step, learning_ratio, r_total_loss, r_total_classification_loss, r_total_localization_loss))

                        if r_total_loss < last_loss:

                            self.save_checkpoint(_global_step, r_total_loss)
                            last_loss = r_total_loss

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

        elif self.optimizer_type == "sgd_optimizer":

            optimizer = tf.train.MomentumOptimizer

        else:
            assert False , "Now Only Support Adam Optimizer"

        return optimizer

    def average_gradients(self, tower_grads):

        average_grads = []

        for grad_and_vars in zip(*tower_grads):
            grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
            grads = tf.concat(grads, 0)
            grad = tf.reduce_mean(grads, 0)
            grad_and_var = (grad, grad_and_vars[0][1])
            average_grads.append(grad_and_var)

        return average_grads


    def average_loss(self, loss_list):

        # tf_loss = tf.concat(loss_list , axis=0)
        tf_loss = tf.stack(loss_list, axis=0)
        mean_loss = tf.reduce_mean(tf_loss)

        return mean_loss