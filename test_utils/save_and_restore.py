# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.model_restore import do_restore_ckpt
from tools.model_restore import do_meta_file_exist
import math
import os

def restore_model(session, saver, check_point_dir):

    do_restore_ckpt(session, saver, check_point_dir)
