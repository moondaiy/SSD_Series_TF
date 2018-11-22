#coding=utf8
import os
import cv2
import numpy as np
import tensorflow as tf


def do_restore_ckpt(sess, saver, checkpoint_dir):

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    try:
        saver.restore(sess, ckpt)
        print('Restore checkpoint from {}'.format(ckpt))
    except Exception as e:
        print(e)
        print("Can not restore from {}".format(checkpoint_dir))
        exit(-1)


def do_meta_file_exist(ckpt_dir):
    fnames = os.listdir(ckpt_dir)
    meta_exists = False
    meta_file_name = ''
    for n in fnames:
        if 'meta' in n:
            meta_exists = True
            meta_file_name = n
            break

    return meta_exists, meta_file_name