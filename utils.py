from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset
import config
import copy
import dgl
import networkx as nx
from tqdm import tqdm

class AugmentedDataset(Dataset):
    def __init__(self, data):
        inputs = [list(reversed(upois)) for upois in data[0]]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])

    def __getitem__(self, index):
        seq, target = self.inputs[index],self.targets[index]
        length = len(seq)

        items = list(np.unique(seq))
        item2id = {n:i for i,n in enumerate(items)}

        graph_data = {
            ('item', 'agg', 'target'):([],[])
        }
        for i in range(config.density):
            graph_data[('item', 'interacts'+str(i), 'item')] = ([],[])

        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros((len(items), 3))
        is_last[item2id[seq[0]]][0] = 1
        is_last[item2id[seq[min(1,len(seq)-1)]]][1] = 1
        is_last[item2id[seq[min(2,len(seq)-1)]]][2] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq if item!= 0]

        for i in range(config.density):
            src, dst = [], []
            for j in range(1, i+2):
                src = src + seq_nid[:-j]
                dst = dst + seq_nid[j:]

            edges = set(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts'+str(i))
               
        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))

        return g, target, length

    def __len__(self):
        return self.length


import itertools

def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


class AugmentingDataset:
    def __init__(self, sessions):
        self.sessions = sessions
        index = create_index(self.sessions)  # columns: sessionId, labelIndex
        self.index = index

    def get28(self):
        fre = defaultdict(int)
        for sess in self.sessions:
            for item in sess:
                fre[item] += 1
        items = [x[0] for x in sorted(fre.items(), key=lambda x: x[1])]
        split = int(len(items) * 0.8)
        return set(items[:split]),  set(items[split:])

    def set28(self, head_items, tail_items):
        self.head_items = head_items
        self.tail_items = tail_items
        
    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq = list(reversed(self.sessions[sid][:lidx]))
        target = self.sessions[sid][lidx]
        length = len(seq)

        
        items = list(np.unique(seq))
        item2id = {n:i for i,n in enumerate(items)}

        graph_data = {
            ('item', 'agg', 'target'):([],[])
        }
        for i in range(config.density):
            graph_data[('item', 'interacts'+str(i), 'item')] = ([],[])
        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros((len(items), 3))
        is_last[item2id[seq[0]]][0] = 1
        is_last[item2id[seq[min(1,len(seq)-1)]]][1] = 1
        is_last[item2id[seq[min(2,len(seq)-1)]]][2] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq]
        
        for i in range(config.density):
            src, dst = [], []
            for j in range(1, i+2):
                src = src + seq_nid[:-j]
                dst = dst + seq_nid[j:]
            
            edges = set(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts'+str(i))

        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))
        return g, target, length

    def __len__(self):
        return len(self.index)

