# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from net.base_net.ssd_net import SSD_Net
from net.data_manager.data_manager import Data_Manager
from configs.configs import parsing_configs
from anchor.Anchor import Anchor
from net.box_utils import boxes_np_op
import cv2
import numpy as np
from test_utils.save_and_restore import restore_model


tf.flags.DEFINE_string('config_path', './configs/ssd_300.yaml', 'config path ')
tf.flags.DEFINE_string('tf_record_path', '/home/tcl/ImageSet/voc/tf_record/train', 'tf record path.')
FLAGS = tf.flags.FLAGS




def render_boxs_info_for_display(image, anchors, labels, scores, encode_box, original_ground_truth, scales, net_out, select_index, net_score, image_size):


    valid_box = net_out[select_index]
    valid_score = net_score[select_index]

    for index, value in enumerate(select_index):

        if net_score[index] > 0.5 and value == True:

            valid_box = net_out[index]
            valid_score = net_score[index]
            original_box = original_ground_truth[index]


            print("current box info is " + str(valid_box))
            print("current box scores is " + str(valid_score))
            # print("current original box info is " + str(box))

            ymin = int(valid_box[0] * image_size)
            xmin = int(valid_box[1] * image_size)
            ymax = int(valid_box[2] * image_size)
            xmax = int(valid_box[3] * image_size)

            # oymin = int(original_box[0] * image_size)
            # oxmin = int(original_box[1] * image_size)
            # oymax = int(original_box[2] * image_size)
            # oxmax = int(original_box[3] * image_size)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), thickness=1,color=(0,0,255))
            # cv2.rectangle(image, (oxmin, oymin), (oxmax, oymax), thickness=1, color=(255, 0, 0))

    return image

def render_rectangle_box(image, box, colour = (255, 255, 255), offset = 0, thickness = 1):
    """
    :param image: 需要显示的图片
    :param box:   box信息
    :param colour: 颜色信息
    :param offset: box偏移
    :param thickness: 线条宽度
    :return:
    """

    height,width, channel = image.shape

    y_start = int(height * box[0]) + offset
    x_start = int(width  * box[1]) + offset

    y_end = int(height * box[2]) + offset
    x_end = int(width  * box[3]) + offset

    image = cv2.rectangle(image,(x_start,y_start), (x_end,y_end), color=colour, thickness= thickness)

    return image

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

            label_out, box_out, score_box, select_index = sess.run(model.finally_box, feed_dict={model.inputs:image_batch , model.is_training: False, model.select_threshold : 0.6, model.nms_threshold : 0.6})

            for i in range(len(image_name_batch)):

                print("-------------------------------------------------------------------------------------")

                image = render_boxs_info_for_display(image_batch[i], np.concatenate(anchor.get_anchors(),axis=0), gt_label_batch[i][:,:21], gt_label_batch[i][:, 25], gt_label_batch[i][:, 21:25], gt_label_batch[i][:, 26:30], anchor_info["prior_scaling"], box_out[i], select_index[i], score_box[i], 300)

                print("-------------------------------------------------------------------------------------")

                cv2.imshow("boxs_info_display", image)
                cv2.waitKey(0)








