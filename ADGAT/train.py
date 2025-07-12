import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ADGAT, AnomalyDetector
from utils import *

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--ad_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--ad_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--ad_dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--th1', type=float, default=0.7, help='threshold_1 for linkpred')
parser.add_argument('--th2', type=float, default=0.9, help='threshold_2 for linkpred')
parser.add_argument('--dataset', type=str, default='cora', help='Graph dataset name',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs'])
parser.add_argument('--attack', type=str, default='meta', 
                    choices=['no', 'meta', 'random', 'nettack', 'hattack'])
parser.add_argument('--ptb', type=float, default=0.05, help="noise ptb_rate",
                    choices=[0, 0.05, 0.1, 0.15, 0.2, 0.25])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if args.cuda else 'cpu')

# 固定seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载所需数据
# 此处获得的是npz文件包含的原始features、adj
origin_feat, origin_adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.ptb, args.attack)
features = normalize_features(origin_feat)
features = torch.FloatTensor(np.array(features.todense()))
adj = normalize_adj(origin_adj + sp.eye(origin_adj.shape[0]))
adj = torch.FloatTensor(np.array(adj.todense()))
n_nodes, feat_dim = features.shape
class_num = int(labels.max()) + 1
detector_param_path = './params/detector_{}_v2.pth'.format(args.dataset)

# 加载异常检测器
detector = AnomalyDetector(args, feat_dim, device)
detector.load_state_dict(torch.load(detector_param_path))
linkpred = detector.infer(origin_feat, origin_adj)  # device = cpu

# 模型训练
model = ADGAT(args, feat_dim, class_num, device)
model.fit(features, adj, labels, idx_train, idx_val, linkpred)

# 模型测试
print("==================================================")
print('Loading last_epoch epoch:')
model.test(features, adj, labels, idx_test, linkpred)

print('Loading best_epoch epoch:')
model.load_state_dict(torch.load('./ADGAT/checkpoint/ckpt_{}_best.pth'.format(args.dataset)))
model.test(features, adj, labels, idx_test, linkpred)
