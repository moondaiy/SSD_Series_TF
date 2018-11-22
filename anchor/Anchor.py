# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from net.box_utils.anchor_np_op import make_anchors_for_all_layer


class Anchor(object):

    def __init__(self, anchor_info, base_info):

        self.base_anchor_sizes = anchor_info["anchor_size"]
        self.anchor_ratios = anchor_info["anchor_ratios"]
        self.feat_shapes = anchor_info["feature_shape"]
        self.anchor_strides = anchor_info["anchor_strides"]
        self.scale_factors = anchor_info["prior_scaling"]
        self.anchor_offset = anchor_info["anchor_offset"]
        self.anchor_pos_iou = anchor_info["anchor_pos_iou_threshold"]
        self.class_numer = base_info["class_number"]
        self.image_shape = base_info["base_net_size"]

        self.anchors, self.total_number = make_anchors_for_all_layer(self.image_shape, self.image_shape, self.base_anchor_sizes, self.anchor_ratios, self.anchor_strides, self.feat_shapes,
                               anchor_offset=self.anchor_offset)


    def get_anchors(self):

        return self.anchors

    def get_anchor_number(self):

        return self.total_number