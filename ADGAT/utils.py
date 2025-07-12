import numpy as np
import scipy.sparse as sp
import torch


def load_data(dataset, ptb, attack):
    """
    加载数据函数，图数据路径保存在"MyPaper/data/"下
    返回值features、adj不做修改, 其余均改为torch.Tensor格式
    """   

    path = "/data1/wangy/Graph/data/{}/{}_{}_{}.npz".format(dataset, dataset, attack, ptb)    # hattack
    if ptb > 0.0:
        ptb = change_ptb(ptb, attack)
        path = "./data/{}/{}_{}_{}.npz".format(dataset, dataset, attack, ptb)
    else:
        path = "./data/{}/{}.npz".format(dataset, dataset)
    
    npz = np.load(path, allow_pickle=True)

    # features、adj不做归一化，直接返回
    adj, features = npz['adj'].item(), npz['features'].item()
    labels = torch.LongTensor(npz['labels'])
    idx_train = torch.LongTensor(npz['idx_split'].item()['idx_train'])
    idx_val = torch.LongTensor(npz['idx_split'].item()['idx_val'])
    idx_test = torch.LongTensor(npz['idx_split'].item()['idx_test'])

    # nettack下，test_nodes = attacked_nodes
    if attack == 'nettack':
        if ptb == 0.0:
            npz = np.load("./data/{}/{}_nettack_1.0.npz".format(dataset, dataset), 
                          allow_pickle=True)
        idx_test = torch.LongTensor(npz['idx_split'].item()['idx_target'])

    print("====================Load the dataset:{} is finished!====================".format(dataset))
    print("====================The dataset root:{}====================".format(path))
    return features, adj, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def change_ptb(ptb, attack):
    # mapping
    meta_nettack = {  
        0.05: 1.0,  
        0.1: 2.0,  
        0.15: 3.0,  
        0.2: 4.0,  
        0.25: 5.0  
    }

    if attack == 'meta':
        return ptb
    elif attack == 'nettack':
        return meta_nettack[ptb]
