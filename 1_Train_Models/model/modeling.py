# %%
from typing import List
from dgl.convert import graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

from model.resnet11 import resnet11

class mpnn(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers):
        super(mpnn, self).__init__()
        self.graphConv = GraphConv(in_feats=in_feats, out_feats=hid_feats, 
                                   weight=True, activation=F.relu)
        self.update = nn.GRU(input_size=hid_feats, hidden_size=in_feats, dropout=0.3)
        self.MLP = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.n_layers = n_layers

    def forward(self, graph, feat):
        m = self.graphConv(graph, feat) # (num_nodes, hid_size)
        # unsqueeze(0)是为了增加seq_len维度, GRU的输入必须要有三维
        _, h = self.update(m.unsqueeze(0), feat.unsqueeze(0)) # (1, num_nodes, input_size)
        h = torch.squeeze(h)    # (num_nodes, input_size)
        for i in range(self.n_layers - 1):
            m = self.graphConv(graph, h)
            _, h = self.update(m.unsqueeze(0), h.unsqueeze(0))
            h = torch.squeeze(h)
        # sigma{MLP(h_v^0, h_v^T)}: h^0和h^T分别经过多层感知机，readout后相加
        graph.ndata['h_0'] = self.MLP(feat)
        graph.ndata['h'] = self.MLP(h)
        res_0 = dgl.readout_nodes(graph, 'h_0') # (num_graphs, out_feats)
        res_T = dgl.readout_nodes(graph, 'h')   # (num_graphs, out_feats)
        return res_0 + res_T

class g2v(nn.Module):
    def __init__(self, in_feats, hid_feats, mpnn_feats, n_layers, out_feats):
        super(g2v, self).__init__()
        self.MPNN = mpnn(in_feats, hid_feats, out_feats=mpnn_feats, n_layers=n_layers)
        self.ResNet = resnet11()
        self.MLP = nn.Sequential(
            nn.Linear(mpnn_feats+32, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, graph, feat, adj_list: List):
        res_above = self.MPNN(graph, feat)  # (batch_size, mpnn_output_size)
        res_below = []  # Expectation: (batch_size, resnet_output_size)
        for i in range(len(adj_list)):
            # 送入ResNet之前加入batch维度
            res_below.append(self.ResNet(adj_list[i].unsqueeze(0)))
        res_below = torch.cat(res_below, 0)
        res = torch.cat([res_above, res_below], 1)
        return self.MLP(res)
        
class siamese(nn.Module):
    def __init__(self, in_feats, hid_feats, mpnn_feats, n_layers, out_feats):
        super(siamese, self).__init__()
        self.g2v = g2v(in_feats, hid_feats, mpnn_feats, n_layers, out_feats)

    def forward(self, graph1, graph2, feats1, feats2, adj_list1, adj_list2):
        output1 = self.g2v(graph1, feats1, adj_list1)
        output2 = self.g2v(graph2, feats2, adj_list2)
        return output1, output2

    def get_func_embedding(self, graph, feats, adj_list):
        output = self.g2v(graph, feats, adj_list)
        return output


# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# adj = g.adj().to_dense() # dgl adj方法返回的是一个稀疏矩阵
# print(adj)
# g = dgl.add_self_loop(g)
# feat = torch.ones(6, 10)
# MPNN = mpnn(10, 32, 20, n_layers=3)
# res = MPNN(g, feat)

# conv = GraphConv(10, 5, norm='both', weight=True, bias=True)
# res = conv(g, feat) # res作为input
# print(res)
# # %%
# rnn = nn.GRU(5, 10, num_layers=1)
# output, hn = rnn(res.unsqueeze(0), feat.unsqueeze(0))   # 添加seq_len维度