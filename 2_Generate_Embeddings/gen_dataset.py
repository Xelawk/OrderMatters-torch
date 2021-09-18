import csv
import os
import sys
import json
import random
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import core
import numpy as np
from dgl.convert import graph as dgl_graph

# 保证在大小目录下运行都能import成功
sys.path.append("1_Train_Models/")
sys.path.append("../1_Train_Models/")
import core.config as config
from core.config import logging
from core.tools import filter_by_arch_opt_levels


block_num_min = config.DATASET_MIN_BLOCK_NUM
block_num_max = config.DATASET_MAX_BLOCK_NUM # sys.maxsize

def gen_func_list(programs):
    # index_uuid: ['0':'coreutilsM/coreutils-8.29_powerpc32_o0', ...]	
    index_uuid = dict()
    index_count = 0
    for program in programs:
        dirs = os.listdir(os.path.join(config.FEA_DIR, program, \
                config.CFG_DFG_GEMINIFEA_VULSEEKERFEA))

        logging.debug('original dirs:{}\n{}'.format(dirs, len(dirs)))

        dirs = [d for d in dirs if filter_by_arch_opt_levels(d)]
        logging.debug('PROGRAMS: {}, ARCHS: {}, OPT_LEVELS: {}'.format( \
                config.PROGRAMS, config.ARCHS, config.OPT_LEVELS))
        logging.debug('filtered dirs:{}\n{}'.format(dirs, len(dirs)))
        for d in dirs:
            index_uuid.setdefault(str(index_count),os.path.join(program,d))
            index_count += 1
        logging.debug('index_uuid: {}'.format(index_uuid))
        logging.debug('index_count: {}'.format(index_count))

    func_list_arr = []
    func_list_dict = defaultdict(list)

    for k, v in tqdm(index_uuid.items()):
        program, v = v.split(os.sep)
        cur_functions_list_file = os.path.join(config.FEA_DIR, program, \
            config.CFG_DFG_GEMINIFEA_VULSEEKERFEA, v, 'functions_list_fea.csv')
        FLAG_FUNCTION_LIST = 0
        if not os.path.exists(cur_functions_list_file):
            cur_functions_list_file = os.path.join(config.FEA_DIR, program, \
                config.CFG_DFG_GEMINIFEA_VULSEEKERFEA, v, 'functions_list.csv')
            FLAG_FUNCTION_LIST = 1
        if not os.path.exists(cur_functions_list_file):
            logging.error('No functions_list.csv in {}'.format(v))
        passCount = 0
        with open(cur_functions_list_file, 'r') as fp:
            logging.debug('Gen dataset: {}'.format(cur_functions_list_file))
            for line in csv.reader(fp):
                if line[0] == '':
                    passCount += 1
                    continue
                if block_num_max > 0:
                    if not (int(line[1]) >= block_num_min and int(line[1]) <= block_num_max):
                            passCount += 1
                            continue
                # TODO: 如果CFG中存在孤岛节点，continue
                graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, program, \
                    config.CFG_DFG_GEMINIFEA_VULSEEKERFEA, v, line[0]+'_cfg.txt'))
                adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=bool))
                edge_list = adj_arr.nonzero()
                cfg = dgl_graph(edge_list)
                if cfg.num_nodes() != len(adj_arr):
                    continue

                if line[0] not in func_list_dict:
                    func_list_arr.append(line[0])
                value = os.path.join(line[3+FLAG_FUNCTION_LIST], \
                        config.CFG_DFG_GEMINIFEA_VULSEEKERFEA, line[4+FLAG_FUNCTION_LIST], line[0])
                func_list_dict[line[0]].append(value)

    print("passCount: " + str(passCount))
    # print(len(func_list_arr))
    logging.debug('len(func_list_arr): {}'.format(len(func_list_arr)))
    random.shuffle(func_list_arr)
    return func_list_arr,func_list_dict

func_list_arr,func_list_dict = gen_func_list(config.PROGRAMS)
func_list_train = []
func_list_valid = []
func_list_test = []
for i in range(len(func_list_arr)):
    if func_list_arr[i] in config.QUERY_FUNC:
        print('QUERY_FUNC is {}'.format(func_list_arr[i]))
        func_list_test.append(func_list_arr[i])
    elif i % 12 == 0:
        func_list_test.append(func_list_arr[i])
    elif i % 12 == 1:
        func_list_valid.append(func_list_arr[i])
    else:
        func_list_train.append(func_list_arr[i])

logging.debug('len(func_list_train(): {}'.format(len(func_list_train)))
logging.debug('len(func_list_valid(): {}'.format(len(func_list_valid)))
logging.debug('len(func_list_test(): {}'.format(len(func_list_test)))

def gen_dataset_search_and_save(func_list_dict, func_list, save_file):
    res_dict = defaultdict(list)
    count = 0
    for func in func_list:
        count += len(func_list_dict[func])
        res_dict[func] = func_list_dict[func]
    with open(save_file, 'w') as fp:
        json.dump(res_dict, fp, indent=4)
    return count

save_file = 'dataset.json'
func_list = func_list_valid + func_list_test + func_list_train

print('number of unique functions: {}'.format(len(func_list)))
total_count = gen_dataset_search_and_save(func_list_dict, func_list, save_file)
print('number of total functions: {}'.format(total_count))