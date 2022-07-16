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
                if length < 10000:
                    hit20.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr20.append(0)
                    else:
                        mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if length < 10000:
                    hit10.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr10.append(0)
                    else:
                        mrr10.append(1 / (np.where(score == target)[0][0] + 1))
            

        result.append(np.mean(hit10) * 100)
        result.append(np.mean(hit20) * 100)
        result.append(np.mean(mrr10) * 100)
        result.append(np.mean(mrr20) * 100)       
        return result


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions

def read_dataset(dataset_dir):
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return test_sessions, num_items


if config.dataset in ['diginetica','gowalla','lastfm']:
    test_sessions, num_items = read_dataset(Path("./dataset/"+config.dataset))
    config.num_node = num_items
    test_data = AugmentingDataset(test_sessions)
else:
    test_data = pickle.load(open('./dataset/' + config.dataset + '/test.txt', 'rb'))
    test_data = AugmentedDataset(test_data)


if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "feat_drop:",config.feat_drop, "label_smooth:",config.lb_smooth, "density:", config.density)


    model = trans_to_cuda(DMGSR(num_node = config.num_node))
    state_dict = torch.load('./checkpoint/'+config.dataset+'.pth')
    model.load_state_dict(state_dict)
    model.eval()
    
    print('-------------------------------------------------------')
    hit10, hit20, mrr10, mrr20 = test(test_data, model=model)
    print('Test Result:')
    print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit10, hit20, mrr10, mrr20))

            