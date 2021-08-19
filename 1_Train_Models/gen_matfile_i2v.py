import numpy as np
import scipy.io
from tqdm import tqdm

import core.config as config
from core.config import logging
from core.tools import bcolors
from core.matfile_core import load_dataset, generate_cfg_pair
from core.matfile_core import generate_feature_pair

def construct_learning_dataset_i2v_binbox(pair_list):
    cfgs_1, cfgs_2 = generate_cfg_pair(pair_list)
    feas_1, feas_2, max_size, nums_1, nums_2 = generate_feature_pair(pair_list, 2)
    return cfgs_1, cfgs_2, feas_1, feas_2, nums_1, nums_2, max_size

def gen_matfile_and_save_i2v_binbox(save_file, pair_list, label_list):
    cfgs_1, cfgs_2, feas_1, feas_2, nums_1, nums_2, max_size = \
                                construct_learning_dataset_i2v_binbox(pair_list)
    # node_list = np.linspace(max_size, max_size, len(label_list), dtype=int)
    # 不清楚node_list有什么用
    # TODO: 获得 G1, G2, F1, F2, label, num1, num2
    # G1, G2 通过填充矩阵得到
    # F1, F2 直接连起来
    assert len(cfgs_1) == len(cfgs_2) == len(feas_1) == len(feas_2) == len(nums_1) == len(nums_2)
    assert (len(feas_1[i]) == nums_1[i] for i in range(len(feas_1)))
    assert (len(feas_2[i]) == nums_2[i] for i in range(len(feas_2)))
    for i in range(len(cfgs_1)):
        cfgs_1[i] = np.pad(cfgs_1[i], ((0,max_size-nums_1[i]),(0,max_size-nums_1[i])), \
            'constant', constant_values=(0,0))
        cfgs_2[i] = np.pad(cfgs_2[i], ((0,max_size-nums_2[i]),(0,max_size-nums_2[i])), \
            'constant', constant_values=(0,0))
    cfgs_1 = np.array(cfgs_1)
    cfgs_2 = np.array(cfgs_2)
    feas_1 = np.concatenate(tuple(feas_1), axis=0).astype(np.float32)
    feas_2 = np.concatenate(tuple(feas_2), axis=0).astype(np.float32)
    label_list = np.array(label_list, dtype=np.int8)
    nums_1.astype(np.int16)     # 减小.mat文件大小
    nums_2.astype(np.int16)
    logging.info('generate tfrecord and save to {}'.format(save_file))
    scipy.io.savemat(save_file, {'G1':cfgs_1, 'G2':cfgs_2, 'F1':feas_1, 'F2':feas_2, \
        'num1':nums_1, 'num2':nums_2, 'label':label_list})
    print(bcolors.OKGREEN, "Generate", save_file, "finished!!", bcolors.ENDC)
    

train_pair, train_label, valid_pair, valid_label, test_pair, test_label = load_dataset()
logging.info('generate matfile: i2v_binbox train...')
gen_matfile_and_save_i2v_binbox(config.MATFILE_I2V_ORDERMATTERS_TRAIN, train_pair, train_label)
logging.info('generate matfile: i2v_binbox valid...')
gen_matfile_and_save_i2v_binbox(config.MATFILE_I2V_ORDERMATTERS_VALID, valid_pair, valid_label)
logging.info('generate matfile: i2v_binbox test...')
gen_matfile_and_save_i2v_binbox(config.MATFILE_I2V_ORDERMATTERS_TEST, test_pair, test_label)
