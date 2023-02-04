import config
import time
import datetime
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import DMGSR
from utils import AugmentedDataset, AugmentingDataset
from pathlib import Path
import pandas as pd
import dgl


def init_seed(seed=None):
    dgl.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def train_test(model, train_data, test_data, epoch, train_sessions):
    print('start training: ', datetime.datetime.now())
    model.train()

    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=20, batch_size=config.batch_size,
                                               shuffle=False, pin_memory=True, collate_fn=collate_fn)
    
    with tqdm(train_loader) as t:
        for bth,data in enumerate(t):
            torch.cuda.empty_cache()
            model.optimizer.zero_grad()
            g, target, _, = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            loss = model(g, targets, training=True)
            


            t.set_postfix(
                        loss = loss.item(),
                        lr = model.optimizer.state_dict()['param_groups'][0]['lr'])
            loss.backward()
            model.optimizer.step()
            total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=20, batch_size=config.batch_size,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
        result = []
        hit20, mrr20 = [], []
        hit10, mrr10 = [], []
        for data in test_loader:
            model.optimizer.zero_grad()
            g, target, length = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores =  model(g, training=False)

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = target.numpy()
            lengths = length.numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if length <= 10000:
                    hit20.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr20.append(0)
                    else:
                        mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, length in zip(sub_scores, targets, lengths):
                if length <= 10000:
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
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items



if config.dataset in ['diginetica','gowalla','lastfm']:
    train_sessions, test_sessions, num_items = read_dataset(Path("./dataset/"+config.dataset))
    config.num_node = num_items
    
    if config.validation:
        num_valid      = int(len(train_sessions) * 0.1)
        test_sessions  = train_sessions[-num_valid:]
        train_sessions = train_sessions[:-num_valid]
    
    train_data = AugmentingDataset(train_sessions)
    test_data = AugmentingDataset(test_sessions)
    
else:
    train_sessions = pickle.load(open('./dataset/' + config.dataset + "/" + '/all_train_seq.txt', 'rb'))
    train_data = AugmentingDataset(train_sessions)
    if config.validation:
        num_valid      = int(len(train_sessions) * 0.1)
        test_sessions  = train_sessions[-num_valid:]
        train_sessions = train_sessions[:-num_valid]
        train_data = AugmentingDataset(train_sessions)
        test_data = AugmentingDataset(test_sessions)
    else:
        train_data = AugmentingDataset(train_sessions)
        test_data = pickle.load(open('./dataset/' + config.dataset + "/" + '/test.txt', 'rb'))
        test_data = AugmentedDataset(test_data)


if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "feat_drop:",config.feat_drop, "label_smooth:",config.lb_smooth, "density:", config.density)
    init_seed(42)


    model = trans_to_cuda(DMGSR(num_node = config.num_node))
    
    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(config.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit10, hit20, mrr10, mrr20 = train_test(model, train_data, test_data, epoch, train_sessions)
        if hit10 >= best_result[0]:
            best_result[0] = hit10
            best_epoch[0] = epoch
        if hit20 >= best_result[1]:
            best_result[1] = hit20
            best_epoch[1] = epoch
        if mrr10 >= best_result[2]:
            best_result[2] = mrr10
            best_epoch[2] = epoch
        if mrr20 >= best_result[3]:
            best_result[3] = mrr20
            best_epoch[3] = epoch
        print('Current Result:')
        print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit10, hit20, mrr10, mrr20))
        print('Best Result:')
        print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3]))
                    
        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))