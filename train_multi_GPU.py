# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from data_manager.data_manager import Data_Manager
from solver.solver_multi_GPU import Solver_multiple_GPU
from configs.configs import parsing_configs
from anchor.Anchor import Anchor


tf.flags.DEFINE_string('config_path', './configs/ssd_dsod_300.yaml', 'config path ')
FLAGS = tf.flags.FLAGS


if __name__=="__main__":

    print("config path is %s "%(FLAGS.config_path))

    # base_info, anchor_info, extract_feature_info, loss_info, train_info
    configs = parsing_configs(FLAGS.config_path)

    base_info             = configs[0]
    anchor_info           = configs[1]
    extract_feature_info  = configs[2]
    loss_info             = configs[3]
    train_info            = configs[4]

    print("config path is %s " % (train_info["tf_record_path"]))

    anchor = Anchor(anchor_info, base_info)

    #配合多GPU训练 data provide 改到内部
    solver = Solver_multiple_GPU(anchor, configs)


    #开始训练
    solver.train()