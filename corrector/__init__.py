#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/tmp/test_py
# Author: Hai Liang Wang
# Date: 2018-03-05:15:13:11
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang<hailiang.hl.wang@gmail.com>, XuMing <xuming624@qq.com>"
__date__      = "2018-03-05:15:13:11"

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

import codecs
import kenlm
import math
import numpy as np
from util import tokenize, preprocess, load_pkl, dump_pkl, any2unicode, any2utf8
from absl import flags   #absl-py
from absl import logging #absl-py
from collections import Counter

FLAGS = flags.FLAGS

'''
model files
'''
################## for language model  ##################
bigram_path = os.path.join(curdir, 'data/kenlm/zhwiki_bigram.klm')
bigram = kenlm.Model(bigram_path)
logging.info('Loaded bigram language model from {}'.format(bigram_path))

trigram_path = os.path.join(curdir,'data/kenlm/zhwiki_trigram.klm')
trigram = kenlm.Model(trigram_path)
logging.info('Loaded trigram language model from {}'.format(trigram_path))

text_path = os.path.join(curdir, 'data/train_input.txt')
text_counter_path = os.path.join(curdir, 'data/train_input_counter.pkl2')
# 字频统计
if os.path.exists(text_counter_path):
    char_counter = load_pkl(text_counter_path)
else:
    logging.info('generate counter from text file:', text_path)
    char_counter = Counter((codecs.open(text_path, 'r', encoding='utf-8').read()))
    dump_pkl(char_counter, text_counter_path)


def _load_same_pinyin(path, sep='\t'):
    """
    加载同音字
    :param path:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        logging.warn("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                same_pron_same_tone = set(list(parts[1]))
                same_pron_diff_tone = set(list(parts[2]))
                value = same_pron_same_tone.union(same_pron_diff_tone)
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value
    return result


def _load_same_stroke(path, sep='\t'):
    """
    加载形似字
    :param path:
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        logging.warn("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 1:
                key_char = parts[0]
                result[key_char] = set(list(parts[1]))
    return result

same_pinyin_text_path = os.path.join(curdir, 'data/same_pinyin.txt')
same_pinyin_model_path = os.path.join(curdir, 'data/same_pinyin.pkl2')
# 同音字
if os.path.exists(same_pinyin_model_path):
    same_pinyin = load_pkl(same_pinyin_model_path)
else:
    logging.info('load same pinyin from text file:', same_pinyin_text_path)
    same_pinyin = _load_same_pinyin(same_pinyin_text_path)
    dump_pkl(same_pinyin, same_pinyin_model_path)

# 形似字
same_stroke_text_path = os.path.join(curdir, 'data/same_stroke.txt')
same_stroke_model_path = os.path.join(curdir, 'data/same_stroke.pkl2')
if os.path.exists(same_stroke_model_path):
    same_stroke = load_pkl(same_stroke_model_path)
else:
    logging.info('load same stroke from text file:', same_stroke_text_path)
    same_stroke = _load_same_stroke(same_stroke_text_path)
    dump_pkl(same_stroke, same_stroke_model_path)


def _get_same_pinyin(char):
    """
    取同音字
    :param char:
    :return:
    """
    return same_pinyin.get(char, set())


def _get_same_stroke(char):
    """
    取形似字
    :param char:
    :return:
    """
    return same_stroke.get(char, set())


def _get_model(n):
    return {2: bigram, 3: trigram, }.get(n, bigram)


def _get_ngram_score(chars, model=bigram):
    return model.score(' '.join(chars), bos=False, eos=False)


def _mad_score(scores, ratio=0.6745):
    """
    平均绝对离差值
    :param ratio:
    :param scores:
    :return: median absolute deviation (MAD) score
    """
    scores = np.array(scores)
    if len(scores.shape) == 1:
        scores = scores[:, None]
    median = np.median(scores, axis=0)  # get median of all scores
    margin_median = np.sqrt(np.sum((scores - median) ** 2, axis=-1))  # deviation from the median
    med_abs_deviation = np.median(margin_median)
    y_score = ratio * margin_median / med_abs_deviation
    return scores, med_abs_deviation, y_score, median


def _get_maybe_error_index(scores, y_score, median, threshold=1.4):
    """
    取疑似错字的位置，通过平均绝对离差（MAD）
    :param scores: np.array
    :param threshold: 阈值越小，得到疑似错别字越多
    :return:
    """
    scores = scores.flatten()
    maybe_error_indices = np.where((y_score > threshold) & (scores < median))
    maybe_error_scores = scores[maybe_error_indices]
    return list(maybe_error_indices[0]), maybe_error_scores


def _overlap(l1, l2):
    """
    检测两个列表中是否有重复值
    :param l1: lists
    :param l2:
    :return:
    """
    if l1[0] < l2[0]:
        if l1[1] <= l2[0]:
            return False
        else:
            return True
    elif l1[0] == l2[0]:
        return True
    else:
        if l1[0] >= l2[1]:
            return False
        else:
            return True


def _overlap_ranges(range1, range2):
    range_set = set()
    for i in range2:
        for j in range1:
            if _overlap(j, i):
                range_set.add(tuple(i))
    return [list(_overlap_range) for _overlap_range in range_set]


def _merge_ranges(ranges):
    """
    合并
    :param ranges:
    :return:
    """
    result = []
    ranges.sort()
    temp = ranges[0][:]
    for start, end in ranges:
        if start <= temp[1]:
            temp[1] = max(temp[1], end)
        else:
            result.append(temp[:])
            temp[0] = start
            temp[1] = end
    result.append(temp[:])
    return result

def _get_frequency(char, counter, total):
    """
    取字符在样本中的词频
    :param char:
    :return:
    """
    return counter[char] / total


def _generate_chars(c, fraction=2):
    """
    取音似、形似字
    :param c:
    :param fraction:
    :return:
    """
    confusion_char_set = _get_same_pinyin(c).union(_get_same_stroke(c))
    if not confusion_char_set:
        confusion_char_set = {c}
    confusion_char_set.add(c)
    confusion_char_list = list(confusion_char_set)
    all_confusion_char = sorted(confusion_char_list, key=lambda k: \
        _get_frequency(k, char_counter, sum(char_counter.values())),
                                reverse=True)
    return all_confusion_char[:len(confusion_char_list) // fraction + 1]


def _correct_chars(sentence, start_index, end_index):
    """
    纠正错字，逐字处理
    :param sentence:
    :param start_index:
    :param end_index:
    :return: corrected characters 修正的汉字
    """
    assert end_index > start_index, "end index must be more than start index"
    chars = sentence[start_index:end_index]
    for i, c in enumerate(chars):
        # 取得所有可能正确的汉字
        maybe_chars = _generate_chars(c)
        logging.debug('num of possible replacements for {} is {}'.format(c, len(maybe_chars)))
        before = sentence[:start_index] + chars[:i]
        after = chars[i + 1:] + sentence[end_index:]
        correct_char = max(maybe_chars, key=lambda k: _get_ngram_score(before + k + after) + math.log(5) ** (k == c))
        chars = chars[:i] + correct_char + chars[i + 1:]
    return chars

'''
Public
'''
def score_sentence(sentence):
    ngram_words = []
    ngram_scores = []
    ngram_avg_scores = []
    for n in [2, 3]:
        words = []
        scores = []
        for i in range(len(sentence) - n + 1):
            word = sentence[i:i + n]
            words.append(word)
            score = _get_ngram_score(word, model=_get_model(n))
            scores.append(score)
        ngram_words.append(words)
        ngrams_scores = list(zip(words, [round(score, 3) for score in scores]))
        logging.debug(ngrams_scores)
        ngram_scores.append(scores)
        # 移动窗口补全得分
        for _ in range(n - 1):
            scores.insert(0, scores[0])
            scores.append(scores[-1])
        avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
        ngram_avg_scores.append(avg_scores)
    # 取拼接后的平均得分
    sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
    scores, mad, y_score, median = _mad_score(sent_scores)
    maybe_error_indices, _ = _get_maybe_error_index(scores, y_score, median)
    merge_range = []
    if maybe_error_indices:
        merge_range = _merge_ranges([[index, index + 1] for index in maybe_error_indices])
    return sent_scores, ngrams_scores, merge_range

def correct(sentence):
    sentence = preprocess(sentence)
    sentence = any2unicode(sentence)
    tokens = tokenize(sentence)
    logging.debug('segment sentens is: %s', ''.join([str(token) for token in tokens]))
    seg_range = [[token[1], token[2]] for token in tokens]
    _, _, maybe_error_range = score_sentence(sentence)
    maybe_error_ranges = []
    if maybe_error_range:
        logging.debug('maybe error range: %s', maybe_error_range)
        maybe_error_ranges = _merge_ranges(_overlap_ranges(maybe_error_range, seg_range))
        for r in maybe_error_ranges:
            start_index, end_index = r
            logging.debug('maybe error words: %s', sentence[start_index:end_index])
            corrected_words = _correct_chars(sentence, start_index, end_index)
            logging.debug('corrected words: %s', corrected_words)
            sentence = sentence[:start_index] + corrected_words + sentence[end_index:]
    return sentence, maybe_error_ranges

import unittest

# run testcase: python /Users/hain/tmp/test_py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_spell(self):
        logging.info("test_spell")
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