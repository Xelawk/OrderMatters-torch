import sys
sys.path.append("1_Train_Models/")
sys.path.append("../1_Train_Models/")
from model.modeling import siamese
import torch
from torch.utils.data import DataLoader
import dgl
from dgl.data import DGLDataset
from dgl.convert import graph as dgl_graph
from scipy import io
import json
from tqdm import tqdm
from collections import defaultdict
import random

import core.config as config

import argparse

parser = argparse.ArgumentParser(description='i2v_binbox')
parser.add_argument('-input', '--input', \
                    help='input matfile', \
                    type=str, \
                    default="dataset.mat")
parser.add_argument('-model', '--model', \
                    help='model path', \
                    type=str, \
                    required=True)
parser.add_argument('-output', '--output', \
                    help='output function embeddings', \
                    type=str, \
                    default="embeddings_lib.json")
parser.add_argument('-query', '--query', \
                    help = 'query file', \
                    type = str, \
                    default='query.json')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

QUERY_NUM = 100

# 模型参数需与训练时的设置保持一致
T = 5 # Message Passing Iteration (default:5)
D = config.WORD2VEC_EMBEDDING_SIZE # dimensional
# H = 3 # num_heads
P = 64 # embedding_size of hidden layers (空间变换) 
M = 64 # mpnn feature size (默认output feature size = mpnn feature size)

class BinBoxEmbedDataset(DGLDataset):
    def __init__(self, force_reload=False, verbose=False):
        super(BinBoxEmbedDataset, self).__init__(name='binbox_embed',
                                                url=None, 
                                                force_reload=force_reload,
                                                verbose=verbose)
    def process(self):
        self.funcnames, self.G, self.F = self._load_graph(args.input)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        funcnames = []
        G = []
        F = []
        func = data['func']
        graphs = data['Graph']
        feats = torch.as_tensor(data['Fea'])
        num = data['num'].squeeze()
        pF = 0 # Fea指针
        print("Building pytorch dataset...")
        for i in tqdm(range(len(num))):
            funcnames.append(func[i].rstrip())
            edge_list = graphs[i].nonzero()
            g = dgl_graph(edge_list)
            G.append(g)
            F.append( feats[pF:pF+num[i]] )
            pF += num[i]
        return funcnames, G, F
    
    def __getitem__(self, idx):
        return self.funcnames[idx], self.G[idx], self.F[idx]
    
    def __len__(self):
        return len(self.funcnames)

dataset = BinBoxEmbedDataset()

def _collate_fn(batch):
    func = [sample[0] for sample in batch]
    graph = [sample[1] for sample in batch]
    feats = [sample[2] for sample in batch]
    adj_list = []
    for i in range(len(graph)):
        G = graph[i].adj().to_dense().unsqueeze(0).to(device)
        adj_list.append(G)
    for i in range(len(graph)):
        graph[i] = dgl.add_self_loop(graph[i])
    g = dgl.batch(graph)
    feats = torch.cat(feats, dim=0) # [node_size, dimensional]
    return func, g, feats, adj_list

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn, drop_last=False)
model = siamese(D, P, mpnn_feats=M, n_layers=T, out_feats=M).to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

func_embedds = defaultdict(list)
count = 0
for i, (funcname, g, feats, adj) in enumerate(tqdm(dataloader)):
    g = g.to(device)
    feats = feats.to(device)
    res = model.get_func_embedding(g, feats, adj)
    # print(res)
    for i in range(len(funcname)):
        func_embedds[funcname[i]] = res[i].tolist()
        count += 1

print("total func: ", count)
with open(args.output, 'w') as fp:
    json.dump(func_embedds, fp, indent=2)

query_funcs = random.sample(func_embedds.keys(), QUERY_NUM)
query_embedds = defaultdict(list)
for func in query_funcs:
    query_embedds[func] = func_embedds[func]
with open(args.query, "w") as wfp:
    json.dump(query_embedds, wfp, indent=2)
