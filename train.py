# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from net.data_manager.data_manager import Data_Manager
from solver.solver import Solver
from configs.configs import parsing_configs
from anchor.Anchor import Anchor


tf.flags.DEFINE_string('config_path', './configs/ssd_300.yaml', 'config path ')
tf.flags.DEFINE_string('tf_record_path', '/home/tcl/ImageSet/voc/tf_record/train', 'tf record path.')
FLAGS = tf.flags.FLAGS







if __name__=="__main__":

    print("config path is %s "%(FLAGS.config_path))
    print("tf record root path is %s" %(FLAGS.tf_record_path))


    # base_info, anchor_info, extract_feature_info, loss_info, train_info
    configs = parsing_configs(FLAGS.config_path)

    base_info             = configs[0]
    anchor_info           = configs[1]
    extract_feature_info  = configs[2]
    loss_info             = configs[3]
    train_info            = configs[4]

    anchor = Anchor(anchor_info, base_info)

    data_provider = Data_Manager(FLAGS.tf_record_path, train_info["batch_size"], base_info["train_step"],
                                 base_info["base_net_size"], anchor.get_anchors(), base_info["class_number"],
                                 anchor_info["prior_scaling"], anchor_info["anchor_pos_iou_threshold"])

    model = SSD_Net(base_info, anchor_info, extract_feature_info, anchor.get_anchors(), loss_info)

    solver = Solver(model, data_provider, anchor.get_anchors(), configs)


    #开始训练
    #
    solver.train()



