# 定义模型、训练模型（train,valid,test）、保存模型
# python train_model_i2v.py [-en exp_name] [-epoch num_epoch]
from sklearn import metrics # sklearn不在torch前面import就会报错
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import dgl
from dgl.data import DGLDataset
from dgl.convert import graph as dgl_graph
from scipy import io
import os
import json
import time
from tqdm import tqdm
import scipy.io

import core.config as config
from core.config import logging
from core.tools import save_dict_to_csv
from model.modeling import siamese

import argparse

parser = argparse.ArgumentParser(description='binbox')

parser.add_argument('-en', '-exp_name', \
                    help='name or goal of this experiment', \
                    type=str, \
                    default='noname')
parser.add_argument('-epoch', '--epoch', \
                    help='train epoch', \
                    type=int, \
                    default=config.NUM_EPOCH)

args = parser.parse_args(args=[])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 5 # Message Passing Iteration (default:5)
D = config.WORD2VEC_EMBEDDING_SIZE # dimensional
# H = 3 # num_heads
P = 64 # embedding_size of hidden layers (空间变换) 
M = 64 # mpnn feature size (默认output feature size = mpnn feature size)
B = 64 # batch size
lr = 0.001
num_epoch = args.epoch

decay_steps = 30
decay_rate = 0.99
snapshot = 1    # validate models between num epoch
display_step = 20 # display between num iter

train_num = config.TRAIN_DATASET_NUM
valid_num = int(train_num / 10.0)
test_num = int(train_num / 10.0)

def calculate_auc(labels, predicts):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predicts, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    # logging.info("auc : {}".format(AUC))
    return AUC

def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.5   # 我觉得这里应该是0才对
    for i in range(len(prediction)):
        if labels[i] == 1:
            if prediction[i] > threshold:
                accu += 1.0
        else:
            if prediction[i] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc

class BinBoxDataset(DGLDataset):
    def __init__(self, DStype=None, force_reload=False, verbose=False):
        self.DStype = DStype    # 这一句必须要写在前面
        super(BinBoxDataset, self).__init__(name='binbox_'+DStype,
                                          url=None,
                                          force_reload=force_reload,
                                          verbose=verbose)
        # 不知道为什么这一句被跳过了
        # self.DStype = DStype

    def process(self):  # 将数据文件转为python数据结构
        if self.DStype == 'train':
            mat_path = config.MATFILE_I2V_ORDERMATTERS_TRAIN
        elif self.DStype == 'valid':
            mat_path = config.MATFILE_I2V_ORDERMATTERS_VALID
            # mat_path = '/home1/mwl/BinBox/data/matfiles/i2v_binbox/valid20000_[3_150]_[i2v_norm5_64]_[openssl]_[all]_[all].mat'
        else:
            mat_path = config.MATFILE_I2V_ORDERMATTERS_TEST
        self.G_1, self.G_2, self.F_1, self.F_2, self.label = \
            self._load_graph(mat_path)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        labels = torch.tensor(data['label']).squeeze()
        G1 = []
        G2 = []
        F1 = []
        F2 = []
        graphs1 = data['G1']
        graphs2 = data['G2']
        feats1 = torch.as_tensor(data['F1'])
        feats2 = torch.as_tensor(data['F2'])
        num1 = data['num1'].squeeze()
        num2 = data['num2'].squeeze()
        pF1 = 0
        pF2 = 0
        # print(len(labels))
        print("Building pytorch " + self.DStype + " dataset...")
        for i in tqdm(range(len(labels))):
            edge_list = graphs1[i].nonzero()
            g = dgl_graph(edge_list)
            G1.append(g)
            edge_list = graphs2[i].nonzero()
            g = dgl_graph(edge_list)
            G2.append(g)
            # 错误出在num
            # print(num1[1])
            F1.append( feats1[ pF1:pF1+num1[i] ] )
            F2.append( feats2[ pF2:pF2+num2[i] ] )
            pF1 += num1[i]
            pF2 += num2[i]
        return G1, G2, F1, F2, labels

    def __getitem__(self, idx):
        return self.G_1[idx], self.G_2[idx], self.F_1[idx], self.F_2[idx], \
            self.label[idx]
    
    def __len__(self):
        return len(self.label)

# 此处需要添加adj_list的生成
def _collate_fn(batch):
    graph1 = [sample[0] for sample in batch]
    graph2 = [sample[1] for sample in batch]
    feats1 = [sample[2] for sample in batch]
    feats2 = [sample[3] for sample in batch]
    labels = [sample[4] for sample in batch]
    adj_list1 = []
    adj_list2 = []
    for i in range(len(graph1)):    # CNN输入有向图，MPNN输入无向图
        # 转成稠密矩阵；添加深度维度；放到GPU上
        G1 = graph1[i].adj().to_dense().unsqueeze(0).to(device)
        G2 = graph2[i].adj().to_dense().unsqueeze(0).to(device)
        adj_list1.append(G1)
        adj_list2.append(G2)
    for i in range(len(graph1)):
        graph1[i] = dgl.add_self_loop(graph1[i])
        graph2[i] = dgl.add_self_loop(graph2[i])
    g1 = dgl.batch(graph1)
    g2 = dgl.batch(graph2)
    feats1 = torch.cat(feats1, dim=0)   # [node_size, dimensional]
    feats2 = torch.cat(feats2, dim=0)
    labels = torch.stack(labels).float() # [batch_size]
    return g1, g2, feats1, feats2, adj_list1, adj_list2, labels

# model供其他文件调用
#model = siamese(D, P, num_heads=H, n_layers=T, feat_drop=0.3, attn_drop=0.3).to(device)
model = siamese(D, P, mpnn_feats=M, n_layers=T, out_feats=M).to(device)

if __name__ == '__main__':
    dataset = BinBoxDataset(DStype='train')
    # first = next(iter(dataset))
    # print(first)
    valid_dataset = BinBoxDataset(DStype='valid')

    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=B, shuffle=True, collate_fn=_collate_fn)
    
    loss_fn = nn.CosineEmbeddingLoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    # 观察模型
    # print(model)
    # for i in model.state_dict():
    #     print(i)
    t0 = time.time()
    for epoch in tqdm(range(num_epoch)):
        total_loss = 0
        total_acc = 0
        for i, (g1, g2, feats1, feats2, adj1, adj2, labels) in enumerate(tqdm(dataloader)):
            g1 = g1.to(device)
            g2 = g2.to(device)
            feats1 = feats1.to(device)
            feats2 = feats2.to(device)
            labels = labels.to(device)
            output1, output2 = model(g1, g2, feats1, feats2, adj1, adj2)
            pred = F.cosine_similarity(output1, output2, dim=-1, eps=1e-6)
            loss = loss_fn(output1, output2, labels)

            prediction = pred.clone().detach()
            acc = compute_accuracy(prediction.cpu(), labels.cpu())
            total_loss += loss.item()
            total_acc += acc
            if (i + 1) % display_step == 0:
                AUC = calculate_auc(labels.cpu(), prediction.cpu())
                print('\nloss:', total_loss/(i+1), ' acc:', total_acc/(i+1), ' auc:', AUC)
            
            if (i + 1) % decay_steps == 0:
                scheduler.step() # 更新学习率

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # valid
        with torch.no_grad():
            valid_count = 0
            valid_loss = 0
            valid_acc = 0
            valid_auc = 0
            for g1, g2, feats1, feats2, adj1, adj2, labels in valid_dataloader:
                g1 = g1.to(device)
                g2 = g2.to(device)
                feats1 = feats1.to(device)
                feats2 = feats2.to(device)
                labels = labels.to(device)
                output1, output2 = model(g1, g2, feats1, feats2, adj1, adj2)
                pred = F.cosine_similarity(output1, output2, dim=-1, eps=1e-6)
                
                loss = loss_fn(output1, output2, labels)
                acc = compute_accuracy(pred.cpu(), labels.cpu())
                AUC = calculate_auc(labels.cpu(), pred.cpu())
                valid_loss += loss.item()
                valid_acc += acc
                valid_auc += AUC
                valid_count += 1
        print('valid loss:', valid_loss/valid_count, ' valid acc:', valid_acc/valid_count, \
            ' valid auc:', valid_auc/valid_count)
        # 保存模型
        # torch.save(model.state_dict(), os.path.join(config.MODEL_BINBOX_DIR, \
        #     'binbox_['+config.PROGRAMS[0]+']_['+config.I2VFEA+']_epoch'+str(epoch)+'.pkl'))
        torch.save(model.state_dict(), os.path.join(config.MODEL_ORDERMATERS_DIR, \
            'orderMatters_' + config.FILENAME_PREFIX + '_epoch' + str(epoch) + '.pkl'))
        print('Cumulative time consumption: ', (time.time()-t0)/60, 'min')
    
    test_dataset = BinBoxDataset(DStype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=True, collate_fn=_collate_fn)
    with torch.no_grad():
        for i, (g1, g2, feats1, feats2, adj1, adj2, labels) in enumerate(test_dataloader):
            g1 = g1.to(device)
            g2 = g2.to(device)
            feats1 = feats1.to(device)
            feats2 = feats2.to(device)
            output1, output2 = model(g1, g2, feats1, feats2, adj1, adj2)
            pred = F.cosine_similarity(output1, output2, dim=-1, eps=1e-6)

            fpr, tpr, _ = metrics.roc_curve(labels, pred.cpu(), pos_label=1)
            fpr = np.array(fpr)
            tpr = np.array(tpr)
            roc_save_file = os.path.join(config.STATIS_ORDERMATTERS_DIR, args.en,
                                'roc-binbox' + config.FILENAME_PREFIX + '_' + \
                                'epoch' + str(num_epoch) + '_' + \
                                'D' + str(D) + '_' + \
                                'T' + str(T) + '_' + \
                                'P' + str(P) + '_' + str(i) + '.mat')
            scipy.io.savemat(roc_save_file, {'fpr':fpr, 'tpr':tpr})