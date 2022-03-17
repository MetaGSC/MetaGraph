import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

import random
import numpy as np

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

_DEBUG = False

dropout_prob = 0.0

class GCN1(MessagePassing):
    def __init__(self, inputfeatures = 2, hidden_channels = 2, num_classes = 2):
        super().__init__(aggr='max') 
        self.mlp = Seq(Linear(2 * inputfeatures, hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x), None

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  
        return self.mlp(tmp)

class GCN(torch.nn.Module):
    def __init__(self, inputfeatures = 2, hidden_channels = [68, 20, 10], num_classes = 2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(inputfeatures, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], num_classes)
        self.classifier = Linear(num_classes, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p = dropout_prob, training=self.training)
        
        x = self.conv2(x, edge_index)   
        out = self.classifier(x)
        return out, x

def train(model, optimizer, criterion, data, train_data):
      model.train()
      optimizer.zero_grad()  
      out, h = model(data.x, data.edge_index)  
      loss = criterion(out[train_data], data.y[train_data])  
      loss.backward()  
      optimizer.step()  
      return loss, h

def iterate(model, optimizer, criterion, data, train_data):
  for epoch in range(1, 101):
      loss = train(model, optimizer, criterion, data, train_data)

def get_parameters(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, criterion

def get_model(inputfeatures = 2, hidden_channels = [69, 20, 10], num_classes = 2):
    model = GCN(inputfeatures = inputfeatures, hidden_channels = hidden_channels, num_classes = num_classes)
    model.double()
    return model

def get_train_validate_test_data(train, validate, index_list):
    size = len(index_list)
    true_list, = np.where(index_list)
    true_count = len(true_list)
    train_set = [False] * size
    validate_set = [False] * size
    test_set = [False] * size

    train_num = int(train * true_count)
    validate_num = int(validate * true_count) 
    test_num = true_count - train_num - validate_num
    true_iterate_index = -1

    for i in range(train_num):
        true_iterate_index += 1
        train_set[true_list[true_iterate_index]] = True

    for i in range(validate_num):
        true_iterate_index += 1
        validate_set[true_list[true_iterate_index]] = True

    for i in range(test_num):
        true_iterate_index += 1    
        test_set[true_list[true_iterate_index]] = True

    return train_set, validate_set, test_set
