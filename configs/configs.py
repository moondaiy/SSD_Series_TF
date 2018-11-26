# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import yaml


def get_value_for_dict(dict ,key):

    return dict.get(key, None)



def parsing_configs(config_path):

    with open(config_path, "r") as f:

        config = yaml.load(f.read())

    if config == None:
        return None

    else:

        base_info = get_value_for_dict(config, "base_info")
        net_name = get_value_for_dict(base_info, "net_name")
        train_step = get_value_for_dict(base_info, "train_step")

        if net_name == "ssd":

            anchor_info = get_value_for_dict(config, "anchor_info")
            extract_feature_info = get_value_for_dict(config, "extract_feature_info")
            loss_info = get_value_for_dict(config, "loss_info")

            if train_step == True:
                infos = get_value_for_dict(config, "training_info")
            else:
                infos = get_value_for_dict(config, "testing_info")

        else:

            raise Exception("Only Support SSD ")

    return base_info, anchor_info, extract_feature_info, loss_info, infos



if __name__=="__main__":

    parsing_configs("ssd_300.yaml")