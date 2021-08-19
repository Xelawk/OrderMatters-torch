import os
import sys
import numpy as np
import csv
import time
import copy
import networkx as nx
import itertools
from tqdm import tqdm

# sys.path.append('/home/simon/zpl/FuncSim')
sys.path.append('/home/xiqian/Binary_similarity/FuncSim/FuncSim')

import core.config as config
from core.config import logging

def load_dataset():
    train_pair, train_label = load_csv_as_pair(config.DATASET_TRAIN)
    valid_pair, valid_label = load_csv_as_pair(config.DATASET_VALID)
    test_pair, test_label = load_csv_as_pair(config.DATASET_TEST)
    return train_pair, train_label, valid_pair, valid_label, test_pair, test_label

def load_csv_as_pair(pair_label_file):
    pair_list = []
    label_list = []
    with open(pair_label_file, 'r') as fp:
        pair_label = csv.reader(fp)
        for line in pair_label:
            pair_list.append([line[0], line[1]])
            label_list.append(int(line[2]))
    return pair_list, label_list

def generate_cfg_pair(pair_list):   # modified
    cfgs_1 = []
    cfgs_2 = []
    logging.info('generate cfg pair...')
    for pair in tqdm(pair_list):
        # logging.debug('pair_1: {}'.format(pair[0]))

        graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, pair[0] + '_cfg.txt'), create_using=nx.DiGraph())
        # logging.debug('num graph nodes: {}'.format(graph_cfg.nodes()))

        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=np.bool))
        # logging.debug(adj_arr)
        # logging.debug('adj_arr: {}'.format(list(itertools.chain.from_iterable(adj_arr))))
        # adj_str = adj_arr.astype(np.string_)
        # logging.debug('adj_str: {}'.format(list(itertools.chain.from_iterable(adj_str))))
        # logging.debug('cfg:{}'.format(b','.join(list(itertools.chain.from_iterable(adj_str)))))

        # cfgs_1.append(b','.join(list(itertools.chain.from_iterable(adj_str))))
        cfgs_1.append(adj_arr)

        graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, pair[1] + '_cfg.txt'))
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=np.bool))
        # adj_str = adj_arr.astype(np.string_)
        # cfgs_2.append(b','.join(list(itertools.chain.from_iterable(adj_str))))
        cfgs_2.append(adj_arr)
    return cfgs_1, cfgs_2

def __generate_cfg_dfg_func(func):
    graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, func + '_cfg.txt'))
    adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
    adj_str = adj_arr.astype(np.string_)
    cfg = b','.join(list(itertools.chain.from_iterable(adj_str)))

    graph_dfg = nx.read_adjlist(os.path.join(config.FEA_DIR, func + '_dfg.txt'))
    
    graph = copy.deepcopy(graph_dfg)
    for node in graph.nodes():
        if not node in graph_cfg:
            graph_dfg.remove_node(node)
    graph_dfg.add_nodes_from(graph_cfg)

    adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_dfg, dtype=float))
    adj_str = adj_arr.astype(np.string_)
    dfg = b','.join(list(itertools.chain.from_iterable(adj_str)))
    
    # logging.debug('cfg:{}'.format(cfg))
    logging.debug('dfg:{}'.format(dfg))
    
    return cfg, dfg

def __generate_cfg_func(func):
    graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, func + '_cfg.txt'))
    adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
    adj_str = adj_arr.astype(np.string_)
    cfg = b','.join(list(itertools.chain.from_iterable(adj_str)))

   # graph_dfg = nx.read_adjlist(os.path.join(config.FEA_DIR, func + '_dfg.txt'))
    
   # graph = copy.deepcopy(graph_dfg)
   # for node in graph.nodes():
   #     if not node in graph_cfg:
   #         graph_dfg.remove_node(node)
   # graph_dfg.add_nodes_from(graph_cfg)

  #  adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_dfg, dtype=float))
  #  adj_str = adj_arr.astype(np.string_)
  #  dfg = b','.join(list(itertools.chain.from_iterable(adj_str)))
    
    # logging.debug('cfg:{}'.format(cfg))
   # logging.debug('dfg:{}'.format(dfg))
    
    return cfg



def generate_cfg_dfg_pair(pair_list):
    cfgs_1 = []
    cfgs_2 = []
    dfgs_1 = []
    dfgs_2 = []
    logging.info('generate cfg & dfg pair...')
    for pair in tqdm(pair_list):
        cfg_1, dfg_1 = __generate_cfg_dfg_func(pair[0])
        cfgs_1.append(cfg_1)
        dfgs_1.append(dfg_1)
        cfg_2, dfg_2 = __generate_cfg_dfg_func(pair[1])
        cfgs_2.append(cfg_2)
        dfgs_2.append(dfg_2)
    return cfgs_1, cfgs_2, dfgs_1, dfgs_2

def __generate_feature_func(func, l, r, flag): # modified
    node_vector = []
    block_feature_dic = {}
    graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, func + '_cfg.txt'), create_using=nx.DiGraph())

    if flag == 2:
        func = func.replace(config.CFG_DFG_GEMINIFEA_VULSEEKERFEA, config.I2VFEA)

    with open(os.path.join(config.FEA_DIR, func + '_fea.csv'), 'r') as fp:
        for line in csv.reader(fp):
            if line[0] == '':
                continue
            block_feature = [float(x) for x in (line[l:r])]
            block_feature_dic.setdefault(str(line[0]), block_feature)

    for node in graph_cfg.nodes():
        node_vector.append(block_feature_dic[node])

    logging.debug('node_feature_size: {}'.format(len(node_vector[0])))
    num_node = len(node_vector)

    node_arr = np.array(node_vector)
    # node_str = node_arr.astype(np.string_)
    # fea = b','.join(list(itertools.chain.from_iterable(node_str)))

    logging.debug('num_node: {}'.format(num_node))
    return num_node, node_arr


def generate_feature_pair(pair_list, flag):
    """
    Args:
        flag:
            0: gemini.
            1: vulseeker.
            2: i2v_***.

    """
    left, right = 1, 8
    if flag == 1:
        left, right = 8,16
    elif flag == 2:
        left, right = 1, config.WORD2VEC_EMBEDDING_SIZE + 1

    feas_1 = []
    feas_2 = []
    nums_1 = []
    nums_2 = []
    nodes_length = []
    logging.info('generate feature pair...')
    for pair in tqdm(pair_list):
        num_node_1, fea_1 = __generate_feature_func(pair[0], left, right, flag)
        feas_1.append(fea_1)
        nums_1.append(num_node_1)
        nodes_length.append(num_node_1)
        num_node_2, fea_2 = __generate_feature_func(pair[1], left, right, flag)
        feas_2.append(fea_2)
        nums_2.append(num_node_2)
        nodes_length.append(num_node_2)
    return feas_1, feas_2, np.max(nodes_length), np.array(nums_1), np.array(nums_2) 

print('xxx')

# test_pair, test_label = load_csv_as_pair(config.DATASET_TEST)
# cfgs_1, cfgs_2, dfgs_1, dfgs_2 = generate_cfg_dfg_pair(test_pair[:10])
# feas_1, feas_2, _, _, _ = generate_feature_pair(test_pair[:10], 2)
