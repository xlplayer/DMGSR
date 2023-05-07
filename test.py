import config
import time
import datetime
import torch
import numpy as np
import pickle
from tqdm import tqdm
from model import DMGSR
from utils import AugmentedDataset, AugmentingDataset
from pathlib import Path
import pandas as pd
import dgl
from collections import defaultdict


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def collate_fn(samples):
    g, target, length = zip(*samples)
    g = dgl.batch(g)
    return g, torch.tensor(target), torch.tensor(length)

def test(test_data, model):
    print('start predicting: ', datetime.datetime.now())
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=20, batch_size=config.batch_size,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
        result = []
        hit20, mrr20 = [], []
        hit10, mrr10 = [], []
        for data in tqdm(test_loader):
            g, target, length = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores = model(g, training=False)
            assert not torch.isnan(scores).any()

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = target.numpy()
            lengths = length.numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if target in test_data.tail_items:
                    hit20.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr20.append(0)
                    else:
                        mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if target in test_data.tail_items:
                    hit10.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr10.append(0)
                    else:
                        mrr10.append(1 / (np.where(score == target)[0][0] + 1))
            

        result.append(np.mean(hit10) * 100)
        result.append(np.mean(hit20) * 100)
        result.append(np.mean(mrr10) * 100)
        result.append(np.mean(mrr20) * 100)  

        hit20s, mrr20s = defaultdict(list), defaultdict(list)
        hit10s, mrr10s = defaultdict(list), defaultdict(list)
        for data in tqdm(test_loader):
            model.optimizer.zero_grad()
            g, target, length = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores =  model(g, training=False)

            # scores = scores[-1] ###mix score
            assert not torch.isnan(scores).any()

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = target.numpy()
            lengths = length.numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if target in test_data.tail_items:
                    hit20s[target].append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr20s[target].append(0)
                    else:
                        mrr20s[target].append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if target in test_data.tail_items:
                    hit10s[target].append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr10s[target].append(0)
                    else:
                        mrr10s[target].append(1 / (np.where(score == target)[0][0] + 1))     

        result.append(np.mean([np.mean(v) for v in hit10s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in hit20s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in mrr10s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in mrr20s.values()]) * 100)                 
        return result


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions

def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items


if config.dataset in ['diginetica','gowalla','lastfm']:
    train_sessions, test_sessions, num_items = read_dataset(Path("./dataset/"+config.dataset))
    config.num_node = num_items
    train_data = AugmentingDataset(train_sessions)
    test_data = AugmentingDataset(test_sessions)
    
else:
    train_sessions = pickle.load(open('./dataset/' + config.dataset + "/" + '/all_train_seq.txt', 'rb'))
    train_data = AugmentingDataset(train_sessions)
    test_data = pickle.load(open('./dataset/' + config.dataset + "/" + '/test.txt', 'rb'))
    test_data = AugmentedDataset(test_data)


if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "feat_drop:",config.feat_drop, "label_smooth:",config.lb_smooth, "density:", config.density)

    tail_items, head_items = train_data.get28()
    test_data.set28(head_items, tail_items)

    model = trans_to_cuda(DMGSR(num_node = config.num_node))
    state_dict = torch.load('./checkpoint/'+config.dataset+'.pth')
    model.load_state_dict(state_dict)
    model.eval()
    
    print('-------------------------------------------------------')
    hit10, hit20, mrr10, mrr20, hit10s, hit20s, mrr10s, mrr20s = test(test_data, model=model)
    print('Test Result:')
    print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit10, hit20, mrr10, mrr20))
    print('\tRecall10s:\t%.4f\tRecalls@20:\t%.4f\tMMRs@10:\t%.4f\tMMRs@20:\t%.4f' % (hit10s, hit20s, mrr10s, mrr20s))

            