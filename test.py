#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/tmp/test_ec.py
# Author: Hai Liang Wang
# Date: 2018-03-05:16:04:39
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-03-05:16:04:39"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

# Get ENV
ENVIRON = os.environ.copy()

from absl import flags   #absl-py
from absl import logging #absl-py

FLAGS = flags.FLAGS
flags.DEFINE_string('echo', None, 'Text to echo.')

from corrector import correct

import unittest

# run testcase: python /Users/hain/tmp/test_ec.py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_convert_pkl(self):
    #     print("test_convert_pkl")
    #     import pickle
    #     for x in ["data/same_pinyin.pkl", 
    #                 "data/same_stroke.pkl",
    #                 "data/train_input_counter.pkl"]:
    #         p = os.path.join(curdir, "corrector", x)
    #         with open(p, "rb") as fin, open("%s.pkl2" % p,"wb") as fout:
    #             w = pickle.load(fin)
    #             pickle.dump(w, fout, protocol=2)

    def test_ec(self):
        logging.info("test_")
        line = '我们现今所使用的大部分舒学符号' # ，你们用的什么婊点符号
        logging.info('input sentence is: %s', line)
        corrected_sent, correct_ranges = correct(line)
        logging.info('corrected_sent: %s', corrected_sent)
        logging.info('correct_ranges: %s', correct_ranges)

def test():
    unittest.main()

if __name__ == '__main__':
    FLAGS([__file__, '--verbosity', '1']) # DEBUG 1; INFO 0; WARNING -1
    test()
