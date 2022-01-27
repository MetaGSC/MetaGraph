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
      # datafolder = "wastewater-plasmidome"
      dataset = os.path.join(directory, datafolder)
      print(dataset, "\n")
      gfafilepath = str(dataset) + "/assembly_graph_with_scaffolds.gfa"
      contigfilepath = str(dataset) + "/contigs.paths"
      fastafilepath = str(dataset) + "/contigs.fasta"

      data, train_data, test_data = generate_data(gfafilepath, contigfilepath, fastafilepath)

      if _DEBUG:
        print("Prepare graphs using data....")

      G = to_networkx(data, to_undirected=True)

      if _DEBUG:
        print("Prepare visualize_graph using data....")

      # visualize_graph(G, color=data.y, filename = "Figures/" + datafolder + "_visualize_graph.png")

      if _DEBUG:
        print("Get the model ....")
      model = get_model(inputfeatures = 2, hidden_channels = 24, num_classes = 2)
      # print(model)

      if _DEBUG:
        print("eval_results of the model ....")
      eval_results = model.eval()
      # print(eval_results)

      if _DEBUG:
        print("Initial Test of the model ....")
      out, h = model(data.x, data.edge_index)
      print(f'Embedding shape: {list(h.shape)}')

      if _DEBUG:
        print("Initial visualize_embedding of the model ....")

      # visualize_embedding(h, color=data.y, filename = "Figures/" + datafolder + "_visualize_embedding.png")
      if _DEBUG:
        print("Initial visualize of the model ....")

      # visualize(h, color=data.y, filename = "Figures/test_visualize.png")

      optimizer, criterion = get_parameters(model)

      if _DEBUG:
        print("Start the training phase....")

      iterate(model, optimizer, criterion, data, train_data)

      if _DEBUG:
        print("Start the testinging phase....")

      test_acc = test(model, data, train_data)
      print("Test Accuracy: ", test_acc)

      model.eval()

      out, h = model(data.x, data.edge_index)
      # visualize(h, color=data.y, filename = "Figures/" + datafolder + "_results_visualize_embedding.png")

      if _DEBUG:
        return data, train_data, test_data
        break
