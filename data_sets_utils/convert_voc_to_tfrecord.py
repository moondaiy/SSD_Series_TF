# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import glob
import numpy as np
import xml.etree.cElementTree as ET
import help_utils

import cv2

#
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('img_dir', 'JPEGImages' , 'img dir')
tf.app.flags.DEFINE_string("VOC_root_dir","/home/tcl/DataBack/voc", "train dir(2012 or 2007)")
tf.app.flags.DEFINE_string("VOC_train_dir","VOC_Train", "train dir(2012 or 2007)")
tf.app.flags.DEFINE_string("VOC_test_dir", "VOC_Test", "test dir(2012 or 2007)")
tf.app.flags.DEFINE_string('train_save_path', '/home/tcl/DataBack/voc/tf_record/train', 'train save name')
tf.app.flags.DEFINE_string('test_save_path',  '/home/tcl/DataBack/voc/tf_record/test', 'train save name')
tf.app.flags.DEFINE_string('tf_train_save_dir', 'train', 'train save dir')
tf.app.flags.DEFINE_string('tf_test_save_dir',  'test', 'train save dir')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('years', 'merge', '2007 2012 merge') #三选择1
FLAGS = tf.app.flags.FLAGS

NAME_LABEL_MAP = {
    'back_ground': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_source_from_flags():

    train_xml_path   = []
    train_image_path = []

    test_xml_path   = []
    test_image_path = []

    image_train_path_2007 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_train_dir, "VOCdevkit", "VOC2007",'JPEGImages')
    image_train_path_2012 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_train_dir, "VOCdevkit", "VOC2012",'JPEGImages')
    image_test_path_2007 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_test_dir, "VOCdevkit", "VOC2007",'JPEGImages')
    image_test_path_2012 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_test_dir, "VOCdevkit", "VOC2012",'JPEGImages')

    xml_train_path_2007 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_train_dir, "VOCdevkit", "VOC2007",'Annotations')
    xml_train_path_2012 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_train_dir, "VOCdevkit", "VOC2012",'Annotations')
    xml_test_path_2007 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_test_dir, "VOCdevkit", "VOC2007",'Annotations')
    xml_test_path_2012 = os.path.join(FLAGS.VOC_root_dir, FLAGS.VOC_test_dir, "VOCdevkit", "VOC2012",'Annotations')

    if FLAGS.years == "merge":

        train_image_path.append(image_train_path_2012)
        train_image_path.append(image_train_path_2007)
        test_image_path.append(image_test_path_2012)
        test_image_path.append(image_test_path_2007)

        train_xml_path.append(xml_train_path_2012)
        train_xml_path.append(xml_train_path_2007)
        test_xml_path.append(xml_test_path_2012)
        test_xml_path.append(xml_test_path_2007)

    elif FLAGS.years == "2012":

        train_image_path.append(image_train_path_2012)
        test_image_path.append(image_test_path_2012)

        train_xml_path.append(xml_train_path_2012)
        test_xml_path.append(xml_test_path_2012)

    elif FLAGS.years == "2007":

        train_image_path.append(image_train_path_2007)
        test_image_path.append(image_test_path_2007)

        train_xml_path.append(xml_train_path_2007)
        test_xml_path.append(xml_test_path_2007)


    return train_image_path, train_xml_path, test_image_path, test_xml_path


def read_xml_gtbox_and_label(xml_path):

    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':

                    xmax = 0
                    ymax = 0
                    xmin = 0
                    ymin = 0

                    tmp_box = []

                    for node in child_item:

                        if node.tag == "xmax":
                            xmax = int(float(node.text))
                        if node.tag == "xmin":
                            xmin = int(float(node.text))
                        if node.tag == "ymax":
                            ymax = int(float(node.text))
                        if node.tag == "ymin":
                            ymin = int(float(node.text))

                    tmp_box.append(xmin)
                    tmp_box.append(ymin)
                    tmp_box.append(xmax)
                    tmp_box.append(ymax)

                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)  # [x1, y1. x2, y2, label]
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, label]

    xmin, ymin, xmax, ymax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2], gtbox_label[:, 3], \
                                    gtbox_label[:, 4]

    gtbox_label = np.transpose(np.stack([ymin, xmin, ymax, xmax, label], axis=0))  # [ymin, xmin, ymax, xmax, label]

    return img_height, img_width, gtbox_label

def do_convert_tf_record(image_path, xml_path, type, record_save_path, count):

    xml_list = glob.glob(xml_path + "/*.xml")

    xml_list_length = len(xml_list)

    tfrecord_saved_path = os.path.join(record_save_path, type + '.tfrecord')

    writer = tf.python_io.TFRecordWriter(path=tfrecord_saved_path)

    #测试
    counter = 0

    for count, xml_file in enumerate(xml_list):

        current_xml_file = xml_file.replace('\\', '/')
        current_image_name = current_xml_file.split('/')[-1].split('.')[0] + FLAGS.img_format
        current_image_path = os.path.join(image_path, current_image_name)

        if not os.path.exists(current_image_path):
            print('%s is not exist!'%(current_image_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml_file)
        img = cv2.imread(current_image_path)

        if gtbox_label.shape[0] > 1:
            print(current_image_name)

        feature = tf.train.Features(feature={
            # maybe do not need encode() in linux
            'img_name': _bytes_feature(current_image_name.encode("utf-8")),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)
        writer.write(example.SerializeToString())

        help_utils.view_bar('Conversion progress', count + 1, xml_list_length)

        #测试
        counter += 1

        if counter >= 10:
            print("%d Test Image is OK ....."%(counter))
            break

def convert_tf_record(step="Train"):

    train_image_path, train_xml_path, test_image_path, test_xml_path = get_source_from_flags()


    if step == "Train":

        for image_path, xml_path in zip(train_image_path, train_xml_path):

            type = xml_path.split('/')[-2]

            do_convert_tf_record(image_path, xml_path, type, FLAGS.train_save_path, 0)

            break

    elif step == "Test":

        for image_path, xml_path in zip(test_image_path, test_xml_path):

            type = xml_path.split('/')[-2]

            do_convert_tf_record(image_path, xml_path, type, FLAGS.test_save_path, 0)






if __name__=="__main__":

    print("Convert Start ...")
    print(FLAGS)

    convert_tf_record("Train")