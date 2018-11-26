# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from net.box_utils.boxes_tf_op import clip_boxes_to_img_boundaries
from tensorflow.python.ops import control_flow_ops


#SSD操作
def sample_distorted_bounding_box_crop(image,labels,bboxes,min_object_covered=0.3,
                                       aspect_ratio_range=(0.9, 1.1),
                                       area_range=(0.80, 1.0),
                                       max_attempts=200):

    def box_translate(bbox_ref, bboxes):

        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])

        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])

        bboxes = bboxes / s

        return bboxes

    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    distort_bbox = distort_bbox[0,0]

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.

    cropped_image.set_shape([None, None, 3])

    #因为图像被crop了因此,原先的bound box坐标也需要进行变换才行
    translate_bboxes = box_translate(distort_bbox, bboxes)


    #对齐边缘.
    translate_bboxes = clip_boxes_to_img_boundaries(translate_bboxes)

    #删除那些经过crop后和过于小的box ,这地方后续进行处理
    # filter_label , filter_box = box_filter_with_iou_for_preprocess(tf.constant([[0,0,1,1]],dtype=translate_bboxes.dtype), translate_bboxes, labels, threshold=0.01, assign_negative=False)

    return cropped_image, labels, translate_bboxes, distort_bbox

def box_info_normilization(gt_box_and_label_tensor, image_height, image_width):

    gt_box_tensor = gt_box_and_label_tensor[:,:4]

    label_tensor =  gt_box_and_label_tensor[:,4]

    ymin = tf.cast(gt_box_tensor[:, 0],tf.float32) / tf.cast(image_height, tf.float32)
    xmin = tf.cast(gt_box_tensor[:, 1],tf.float32) / tf.cast(image_width , tf.float32)

    ymax = tf.cast(gt_box_tensor[:, 2],tf.float32) / tf.cast(image_height, tf.float32)
    xmax = tf.cast(gt_box_tensor[:, 3],tf.float32) / tf.cast(image_width , tf.float32)

    stack_tensor = tf.stack([ymin, xmin, ymax, xmax], axis= 1)

    label_tensor = tf.reshape(tf.cast(label_tensor,tf.float32),(-1,1))

    convert_gt_and_label_tensor = tf.concat([stack_tensor, label_tensor], axis = 1)

    return convert_gt_and_label_tensor

def resize_image_with_fixed_size(image_tensor, height, width, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):

    size = (height, width)

    resize_image_tensor = tf.image.resize_images(image_tensor, size, method, align_corners)

    return resize_image_tensor


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''
    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w//h),
                           false_fn=lambda: (target_shortside_len * h//w,  target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)

    xmin, xmax = xmin * new_w//w, xmax * new_w//w
    ymin, ymax = ymin * new_h//h, ymax * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0) # ensure imgtensor rank is 3
    return img_tensor, tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                               false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]


def flip_left_right(img_tensor, gtboxes):

    img_tensor = tf.image.flip_left_right(img_tensor)

    ymin, xmin, ymax, xmax = tf.unstack(gtboxes, axis=1)

    new_xmin = 1.0 - xmax
    new_xmax = 1.0 - xmin

    # return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax], axis=0))

#随机左右镜像操作
def random_flip_left_right(img_tensor, gtboxes_tensor):

    img_tensor, gtboxes_tensor = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_tensor),
                                            lambda: (img_tensor, gtboxes_tensor))

    return img_tensor,  gtboxes_tensor


def apply_with_random_selector(x, func, process_number):

    sel = tf.random_uniform([], maxval=process_number, dtype=tf.int32)

    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(process_number)])[0]

def random_adjust_image_pixes(image_tensor):

    def distort_color(image, color_ordering=0, fast_mode=True):

        if fast_mode:

            if color_ordering == 0:

                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

            return tf.clip_by_value(image, 0.0, 1.0)

    return apply_with_random_selector(image_tensor, lambda x, ordering: distort_color(x, ordering, False), 4)


def convert_image_format(image_tensor, type = "0_1"):

    if type == "0_1":

        image_tensor = tf.cast(image_tensor, dtype=tf.float32) / 255.0

    elif type == "0_255":

        image_tensor = tf.cast(image_tensor, dtype=tf.float32) * 255.0

    else:

        raise ValueError('The supported type is 0_1  0_255')

    return image_tensor


#VGG mean value
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def image_whitened(image_tensor, means = [_R_MEAN, _G_MEAN, _B_MEAN]):

    if image_tensor.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    mean = tf.constant(means, dtype=image_tensor.dtype)

    image_tensor = image_tensor - mean

    return image_tensor






