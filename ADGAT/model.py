import time
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from layer import *
from utils import accuracy


class ADGAT(nn.Module):
    def __init__(self, args, nfeat, nclass, device):
        """Dense Version."""
        super(ADGAT, self).__init__()
        self.device = device
        self.args = args
        self.dataset = args.dataset
        self.nhid=args.hidden
        self.nheads=args.nb_heads
        self.alpha=args.alpha
        self.dropout = args.dropout
        self.th1 = args.th1
        self.th2 = args.th2
        self.lr=args.lr
        self.weight_decay=args.weight_decay
        
        # self.attentions = [GraphAttentionLayer(nfeat, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        self.attentions = []
        for _ in range(self.nheads):
            attention_layer = GraphAttentionLayer(nfeat, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True)
            self.attentions.append(attention_layer)

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, nclass, dropout=self.dropout, alpha=self.alpha, concat=False)
        self.to(self.device) # 将模型移动到设备上

    def fit(self, features, adj, labels, idx_train, idx_val, linkpred=None):
        args = self.args
        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        linkpred = linkpred.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_val = idx_val.to(self.device)

        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr, 
                               weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        t_total = time.time()
        bad_counter = 0
        best_epoch = 0
        best = args.epochs + 1
        for epoch in range(args.epochs):
            self.train()
            t = time.time()
            optimizer.zero_grad()
            output = self(features, adj, linkpred)
            loss_train = criterion(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            acc_train = accuracy(output[idx_train], labels[idx_train])
            
            # 验证集评估
            self.eval()
            output = self(features, adj, linkpred)
            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            # 基于loss_val的早停
            if loss_val < best:
                best = loss_val
                best_epoch = epoch
                bad_counter = 0
                torch.save(self.state_dict(), './ADGAT/checkpoint/ckpt_{}_best.pth'.format(self.dataset))
            else:
                bad_counter += 1
            
            if bad_counter == args.patience:
                print("No optimization for a long time, auto-stopping...")
                break

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def test(self, features, adj, labels, idx_test, linkpred):
        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        linkpred = linkpred.to(self.device)
        idx_test = idx_test.to(self.device)

        self.eval()
        output = self(features, adj, linkpred)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))

    def forward(self, x, adj, linkpred=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, self.th1, self.th2, linkpred) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, self.th1, self.th2, linkpred))
        return F.log_softmax(x, dim=1)


class AnomalyDetector(nn.Module):
    def __init__(self, args, feat_dim, device):
        super(AnomalyDetector, self).__init__()
        self.device = device
        self.args = args
        self.hidden_dim1 = args.ad_hidden1
        self.hidden_dim2 = args.ad_hidden2
        self.dropout = args.ad_dropout
        self.gc1 = GraphConvolution(feat_dim, self.hidden_dim1, self.dropout, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim1, self.hidden_dim2, self.dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(self.hidden_dim1, self.hidden_dim2, self.dropout, act=lambda x: x)
        self.dc = AnomalyDetectorDecoder(self.hidden_dim2, self.dropout, act=lambda x: x)
        self.to(self.device)  # 将模型移动到设备上

    def infer(self, x, a):
        # 禁用model参数梯度，参数梯度不参与计算/更新
        for param in self.parameters():
            param.requires_grad = False

        # 推理输入图=(X, A)
        x = torch.tensor(x.toarray())
        a = self.preprocess_graph(a)
        x = x.to(self.device)
        a = a.to(self.device)

        # 禁用梯度计算，不生成计算图
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x, a)
            z = self.reparameterize(mu, logvar)
            z = self.dc(z).sigmoid()
        return z.cpu()

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
    
    # VAE重参数化操作, 不改变shape
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def preprocess_graph(self, adj):
        """normalization adj"""
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    