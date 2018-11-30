# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from data_manager.data_manager import   Data_Manager
from configs.configs import parsing_configs
from anchor.Anchor import Anchor
import cv2
from test_utils.save_and_restore import restore_model
from test_utils.display_tools import render_boxs_info_for_display
import time
import numpy as np



tf.flags.DEFINE_string('config_path', './configs/ssd_dsod_300.yaml', 'config path ')
tf.flags.DEFINE_string('tf_record_path', '/home/tcl/ImageSet/voc/tf_record/test', 'tf record path.')
FLAGS = tf.flags.FLAGS



if __name__=="__main__":

    print("config path is %s "%(FLAGS.config_path))

    # base_info, anchor_info, extract_feature_info, loss_info, train_info
    configs = parsing_configs(FLAGS.config_path)

    base_info             = configs[0]
    anchor_info           = configs[1]
    extract_feature_info  = configs[2]
    loss_info             = configs[3]
    test_info             = configs[4]

    anchor = Anchor(anchor_info, base_info)

    data_provider = Data_Manager(test_info["tf_record_path"], test_info["batch_size"], base_info["train_step"],
                                 base_info["base_net_size"], anchor.get_anchors(), base_info["class_number"],
                                 anchor_info["prior_scaling"], anchor_info["anchor_pos_iou_threshold"])

    model = SSD_Net(base_info, anchor_info, extract_feature_info, anchor.get_anchors(), loss_info)

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

    with model.session as sess:

        sess.run(init_op)

        restore_model(sess, saver, "./checkpoint_dir")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_number in range(3):



            image_name_batch, image_batch, gt_label_batch, num_object, img_height, img_width = \
                sess.run((data_provider.next_batch()))

            start_time = time.clock()

            label_out, box_out, score_box, select_index = sess.run(model.finally_box, feed_dict={model.inputs:image_batch , model.is_training: False, model.select_threshold : 0.5, model.nms_threshold : 0.5})

            end_time = time.clock()

            print("Time is %f"%(end_time - start_time))

            for i in range(len(image_name_batch)):

                print("------------------------------%s Start ----------------------------------------------"%(image_name_batch[i]))

                image = render_boxs_info_for_display(image_batch[i], box_out[i], select_index[i], score_box[i], base_info["base_net_size"])

                print("------------------------------%s End--------------------------------------------------"%(image_name_batch[i]))

                cv2.imshow("boxs_info_display", image.astype(np.uint8))
                cv2.waitKey(0)








