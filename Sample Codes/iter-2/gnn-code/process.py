import os
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from prepare import generate_data
from visualization import visualize_graph, visualize_embedding, visualize
from gnn_model import GCN, get_parameters, get_model, iterate, test

gnndatasetpath = "/home/hp/FYP/GNN/gnn-datasets/Test_Datasets"
_DEBUG = False

def process():
  directory = gnndatasetpath
  data = None
  for datafolder in os.listdir(directory):
      dataset = os.path.join(directory, datafolder)
      print(dataset, "\n")
      gfafilepath = str(dataset) + "/assembly_graph_with_scaffolds.gfa"
      contigfilepath = str(dataset) + "/contigs.paths"
      fastafilepath = str(dataset) + "/contigs.fasta"

      data, train_data, test_data = generate_data(gfafilepath, contigfilepath, fastafilepath)

      G = to_networkx(data, to_undirected=True)
      visualize_graph(G, color=data.y, filename = "Figures/" + datafolder + "_visualize_graph.png")

      model = get_model(inputfeatures = 2, hidden_channels = 24, num_classes = 2)
      # print(model)

      eval_results = model.eval()
      # print(eval_results)

      out, h = model(data.x, data.edge_index)
      print(f'Embedding shape: {list(h.shape)}')

      visualize_embedding(h, color=data.y, filename = "Figures/" + datafolder + "_visualize_embedding.png")
      visualize(h, color=data.y, filename = "Figures/test_visualize.png")

      optimizer, criterion = get_parameters(model)

      iterate(model, optimizer, criterion, data, train_data)

      test_acc = test(model, data, train_data)
      print("Test Accuracy: ", test_acc)

      model.eval()

      out, h = model(data.x, data.edge_index)
      visualize(h, color=data.y, filename = "Figures/" + datafolder + "_results_visualize_embedding.png")


      if _DEBUG:
        return data, train_data, test_data
        break