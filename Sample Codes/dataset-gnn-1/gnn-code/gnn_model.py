import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

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
      test_acc = int(test_correct.sum()) / int(test_data.sum())  # Derive ratio of correct predictions.
      return test_acc, h

def iterate(model, optimizer, criterion, data, train_data):
  for epoch in range(1, 101):
      loss = train(model, optimizer, criterion, data, train_data)
      if _DEBUG:
        print(f'Epoch: ', epoch, "Loss: ", loss)

def get_parameters(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, criterion

def get_model(inputfeatures = 2, hidden_channels = 24, num_classes = 2):
    model = GCN(inputfeatures = 2, hidden_channels = 24, num_classes = 2)
    model.double()
    return model
