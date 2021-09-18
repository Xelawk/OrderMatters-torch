import csv
import os
import sys
import json
import glob
import random
import numpy as np
import scipy.io
from tqdm import tqdm
from collections import defaultdict
import torch

# 保证在大小目录下运行都能import成功
sys.path.append("1_Train_Models/")
sys.path.append("../1_Train_Models/")
import core.config as config
from core.config import logging
from core.tools import filter_by_arch_opt_levels
from core.matfile_core import __generate_cfg_func, __generate_feature_func

import argparse

parser = argparse.ArgumentParser(description='gen_matfile')
parser.add_argument('-input', '--input', \
                    help='input dataset file', \
                    type=str, \
                    default="dataset.json")
parser.add_argument('-output', '--output', \
                    help='output matfile file', \
                    type=str, \
                    default="dataset.mat")

args = parser.parse_args()

block_num_min = config.DATASET_MIN_BLOCK_NUM
block_num_max = config.DATASET_MAX_BLOCK_NUM

def gen_matfile_search_and_save(func_list_dict, func_list, save_file):
    funcnames = []
    cfgs = []
    feas = []
    nums = []

    count = 0

    for func in tqdm(func_list):
        count += len(func_list_dict[func])
        for unit in func_list_dict[func]:
            cfg = __generate_cfg_func(unit)
            num, fea = __generate_feature_func(unit, 1, config.WORD2VEC_EMBEDDING_SIZE+1, flag=2)
            funcnames.append(unit)
            cfgs.append(cfg)
            feas.append(fea.astype(np.float32))
            nums.append(num)
    num_max = max(nums)
    assert (len(feas[i]) == nums[i] for i in range(len(feas)))
    for i in range(len(cfgs)):
        cfgs[i] = np.pad(cfgs[i], ((0,num_max-nums[i]),(0,num_max-nums[i])), \
            'constant', constant_values=(0,0))
    cfgs = np.array(cfgs)
    feas = np.concatenate(tuple(feas), axis=0)
    funcnames = np.array(funcnames)
    nums = np.array(nums).astype(np.int16)
    scipy.io.savemat(save_file, {'func':funcnames, 'Graph': cfgs, 'Fea':feas, 'num':nums})
    # node_list = np.linspace(block_num_max, block_num_max, len(funcnames), dtype = int)
    return count

with open(args.input, 'r') as fp:
    func_list_dict = json.load(fp)

save_file = args.output
print('number of unique functions: ', len(func_list_dict.keys()))
total_count = gen_matfile_search_and_save(func_list_dict, func_list_dict.keys(), save_file)
print('number of total functions: ', total_count)