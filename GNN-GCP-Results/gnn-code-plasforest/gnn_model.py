import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

import random

_DEBUG = False

class GCN(torch.nn.Module):
    def __init__(self, inputfeatures = 2, hidden_channels = 2, num_classes = 2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(inputfeatures, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.classifier = Linear(num_classes, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # Apply a final (linear) classifier.
        out = self.classifier(x)
        return out, x

def train(model, optimizer, criterion, data, train_data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[train_data], data.y[train_data])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss, h

def test(model, data, test_data):
      model.eval()
      out, h = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      # test_correct = pred[test_data] == data.y[test_data]  # Check against ground-truth labels.
      test_correct = pred[test_data] == data.y[test_data]  # Check against ground-truth labels.
      test_acc = float(test_correct.sum()) / float(test_data.sum())  # Derive ratio of correct predictions.
      return test_acc, h

def iterate(model, optimizer, criterion, data, train_data, validate_set):
  for epoch in range(1, 101):
      loss = train(model, optimizer, criterion, data, train_data)
      test_acc, h = test(model, data, validate_set)
      if _DEBUG or True:
    #     validate_len = len(validate_set)
    #     train_len = len(train_data)
        #   print(f'Epoch: ', epoch, " Validation accuracy ", test_acc, validate_len, train_len)
          print(f'Epoch: ', epoch, " Validation accuracy ", test_acc)

def get_parameters(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, criterion

def get_model(inputfeatures = 2, hidden_channels = 24, num_classes = 2):
    model = GCN(inputfeatures = inputfeatures, hidden_channels = hidden_channels, num_classes = num_classes)
    model.double()
    return model

def get_train_validate_test_data(train, validate, index_list):
    size = len(index_list)
    true_count = 0
    for value in index_list:
        if (value):
            true_count += 1

    train_set = [False] * size
    validate_set = [False] * size
    test_set = [False] * size

    train_num = int(train * true_count)
    validate_num = int(validate * true_count) 
    test_num = true_count - train_num - validate_num

    # random.shuffle(list(index_list))

    for i in range(size):
        if (index_list[i] and train_num > 0):
            train_set[i] = True
            train_num -= 1
        elif (index_list[i] and validate_num > 0):
            validate_set[i] = True
            validate_num -= 1
        elif (index_list[i] and test_num > 0):
            test_set[i] = True
            test_num -= 1

    return train_set, validate_set, test_set