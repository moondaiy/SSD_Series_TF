# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from configs.configs import parsing_configs
from net.box_utils import anchor_np_op
from net.box_utils import boxes_np_op
from data_manager import data_manager
from net.base_net import ssd_net
import cv2


np.set_printoptions(suppress = True)

def render_boxs_info_for_display(image, anchors, labels, scores, encode_box, original_ground_truth, scales, net_out, anchor_pos_iou = 0):

    for index, score in enumerate(scores):

        # print("current score is %f" % (score))
        # print("current label :" + str(labels[index]))

        if score > 0:

            print("encode gt box is  " + str(encode_box[index]))
            # print("net out encode is " + str(net_out[index]))

            decode_box = boxes_np_op.decode_boxes(np.expand_dims(encode_box[index],axis=0), np.expand_dims(anchors[index], axis=0), scales)

            print("current score is %f"%(score))
            print("anchor box : " + str(anchors[index]))
            print("decode gt box : " + str(decode_box))
            print("original gt box" + str(original_ground_truth[index]))
            print("current label :" + str(labels[index]))

            image = render_rectangle_box(image, anchors[index], colour=(0, 0, 255))
            image = render_rectangle_box(image, decode_box[0], colour=(0, 255, 0))
            image = render_rectangle_box(image, original_ground_truth[index], colour=(255, 0, 0), offset = 3, thickness=2)

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

    y_start = int(height * box[0]) + offset if int(height * box[0]) + offset > 0 else 0
    x_start = int(width  * box[1]) + offset if int(width  * box[1]) + offset > 0 else 0

    y_end = int(height * box[2]) + offset if int(height * box[2]) + offset <  height else height - 1
    x_end = int(width  * box[3]) + offset if int(width  * box[3]) + offset <  width  else width - 1

    image = cv2.rectangle(image,(x_start,y_start), (x_end,y_end), color=colour, thickness= thickness)

    return image


def judge_data_valid(image_name , gt_box_label, image =None, extra = None):

    if np.any(np.isnan(gt_box_label)) == True:

        nan_index = np.where(np.isnan(gt_box_label))

        print("current image name is %s"%(image_name))
        print(gt_box_label[nan_index[0],:])

        print("***************************************")
        if extra is not None:
            print(extra)

        current_gt_label = gt_box_label[nan_index[0],:]

        if image is not None:

            image = render_rectangle_box(image, current_gt_label[:, 26:30][0], colour=(255, 0, 0), offset=0, thickness=2)
            cv2.imshow("vliad", image)
            cv2.waitKey()




        # print(nan_index)



if __name__=="__main__":


    config_path = "./configs/ssd_300.yaml"
    tf_record_path = "/home/tcl/ImageSet/voc/tf_record/train"
    configs = parsing_configs(config_path)

    base_net_info = configs[0]
    anchor_info = configs[1]
    extract_feature_info = configs[2]
    loss_info = configs[3]

    ground_truth = np.array([[0.25, 0.25, 0.5,0.5, 0],
                             [0.5, 0.5, 0.75, 0.75, 3],
                             [0.3, 0.3, 0.7 , 0.7,  4]])

    ground_truth_tensor = tf.convert_to_tensor(ground_truth,dtype=tf.float32)

    base_anchor_sizes = anchor_info["anchor_size"]
    anchor_ratios     = anchor_info["anchor_ratios"]
    feat_shapes       = anchor_info["feature_shape"]
    anchor_strides    = anchor_info["anchor_strides"]
    scale_factors     = anchor_info["prior_scaling"]
    anchor_offset     = anchor_info["anchor_offset"]
    anchor_pos_iou    = anchor_info["anchor_pos_iou_threshold"]
    class_numer       = base_net_info["class_number"]
    image_shape       = base_net_info["base_net_size"]
    is_training       = base_net_info["train_step"]

    total_number = 0

    #ymin xmin ymax xmax
    all_anchors, total_number = anchor_np_op.make_anchors_for_all_layer(image_shape, image_shape, base_anchor_sizes, anchor_ratios, anchor_strides, feat_shapes,
                               anchor_offset=anchor_offset)


    batch_size = 16
    image_size = 300

    data_provider  = data_manager.Data_Manager(tf_record_path, batch_size, is_training, image_size, all_anchors, class_numer, scale_factors, anchor_pos_iou)

    net = ssd_net.SSD_Net(base_net_info, anchor_info, extract_feature_info, all_anchors, loss_info)

    print("总共的anchor个数为: %d"%(total_number))

    all_anchors = np.concatenate(all_anchors, axis=0)

    all_anchors_tensor = tf.constant(all_anchors, dtype=tf.float32)

    total_localization_loss, total_classification_loss, total_loss = net.build_loss(net.multibox_layer_out, net.labels, net.total_anchor_number)

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_number in range(100):

            print("*****************************************************************************")
            image_name_batch, image_batch, gt_label_batch, num_object, img_height, img_width = \
                sess.run((data_provider.next_batch()))

            for i in range(len(image_name_batch)):

                print("-----------------------------%s   display start-------------------------------------------------"%(image_name_batch[i]))

                image = render_boxs_info_for_display(image_batch[i], all_anchors, gt_label_batch[i][:,:21], gt_label_batch[i][:, 25], gt_label_batch[i][:, 21:25], gt_label_batch[i][:, 26:30], scale_factors, None, anchor_pos_iou)

                print("------------------------------%s  display end -------------------------------------------------------"%(image_name_batch[i]))

                cv2.imshow("boxs_info_display", image.astype(np.uint8))
                cv2.waitKey(0)
                judge_data_valid(image_name_batch[i], gt_label_batch[i], image_batch[i])

                # r_total_localization_loss, r_total_classification_loss, r_total_loss, r_localization  = sess.run([total_localization_loss, total_classification_loss, total_loss , net.net_out],feed_dict={net.labels :gt_label_batch, net.inputs : image_batch , net.is_training:True})

            # for i in range(len(image_name_batch)):
            #
            #     print("-------------------------------------------------------------------------------------")
            #
            #     image = render_boxs_info_for_display(image_batch[i], all_anchors, gt_label_batch[i][:,:21], gt_label_batch[i][:, 25], gt_label_batch[i][:, 21:25], gt_label_batch[i][:, 26:30], scale_factors, r_localization[i][:,21:25], anchor_pos_iou)
            #
            #     print("-------------------------------------------------------------------------------------")
            #
            #     cv2.imshow("boxs_info_display", image)
            #     cv2.waitKey(0)



                # print("localization loss is %f   classification loss  is %f  total loss is %f"%(r_total_localization_loss, r_total_classification_loss, r_total_loss))




