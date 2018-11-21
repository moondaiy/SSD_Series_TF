# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from net.base_net.ssd_net import SSD_Net
from configs.configs import parsing_configs




if __name__=="__main__":

    config_path = "./configs/ssd_300.yaml"

    configs = parsing_configs(config_path)

    base_net_info = configs[0]
    anchor_info = configs[1]
    extract_feature = configs[2]

    net = SSD_Net(base_net_info, anchor_info, extract_feature)





