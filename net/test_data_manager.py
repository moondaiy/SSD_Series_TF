# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from net.data_manager.data_manager import Data_Manager
import cv2

if __name__ == "__main__":

    tf_record_path = "/home/tcl/ImageSet/voc/tf_record/train"
    batch_size = 5
    is_training = True
    image_size = 300

    data_provider  = Data_Manager(tf_record_path, batch_size, is_training, image_size)

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_name_batch, image_batch, gt_label_batch, num_object, img_height, img_width, image_with_box = sess.run(data_provider.next_batch())

        for i in range(batch_size):

            image_display = image_with_box[i]
            image = image_batch[i]
            image_name = str(image_name_batch[i])

            gt = gt_label_batch[i]
            print(gt)
            # print(image_name_batch[i])
            # print(img_height[i])
            # print(img_width[i])

            # cv2.imshow(image_name + "_box", image_display.astype(np.uint8))
            # cv2.imshow(image_name, image.astype(np.uint8))

            cv2.imshow(image_name + "_box", image_display)
            cv2.imshow(image_name, image)

            cv2.waitKey(0)