import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class AnomalyDetectorDecoder(nn.Module):
    """
    异常检测器的decoder端
    """
    def __init__(self, encoder_dim, dropout, act, bias=True):
        super(AnomalyDetectorDecoder, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(encoder_dim, 32, bias=bias)
        self.fc2 = nn.Linear(32, 16, bias=bias)
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        h1 = torch.relu(self.fc1(z))
        h2 = self.fc2(h1)
        adj = self.act(torch.mm(h2, h2.t()))
        return adj
    

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Initialize the eta as parameters
        self.eta1 = nn.Parameter(torch.tensor(-9e15))
        self.eta2 = nn.Parameter(torch.tensor(1.0))     
        self.eta3 = nn.Parameter(torch.tensor(3.0))

        self.eta = nn.Parameter(torch.tensor(1.0))  # add

    def forward(self, h, adj, th1, th2, linkpred=None):
        # shape: h = (N, in_features)
        # shape: Wh = (N, out_features)
        Wh = torch.mm(h, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = self.eta1 * torch.ones_like(e)
        if linkpred is not None:
            linkpred[adj == 0] = 0
            attention = torch.where(linkpred <= th1, zero_vec, e)
            attention = torch.where((linkpred > th1) & (linkpred < th2), 
                                    self.eta2 * e, attention)
            attention = torch.where(linkpred >= th2, self.eta3 * e, attention)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
            
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'