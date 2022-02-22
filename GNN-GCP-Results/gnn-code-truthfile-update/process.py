import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from prepare import get_the_segment_contig_map, generate_edge_tensor
from visualization import visualize_graph, visualize_embedding, visualize
from gnn_model import GCN, get_parameters, get_model, iterate, test, get_train_validate_test_data
from preprocess import preprocess

gnndatasetpath = "/home/hp/FYP/GNN/gnn-datasets/Test_Datasets"
csvfiles = "/home/hp/FYP/GNN/gnn-code-plasforest/plasforestresults"
_DEBUG = False

def process():
  directory = gnndatasetpath
  data = None
  for datafolder in os.listdir(directory):
      # datafolder = "wastewater-plasmidome"
      dataset = os.path.join(directory, datafolder)
      gfafilepath = str(dataset) + "/assembly_graph_with_scaffolds.gfa"
      contigfilepath = str(dataset) + "/contigs.paths"
      fastafilepath = str(dataset) + "/contigs.fasta"
      truthfilepath = str(dataset) + "/truth"

      segment_contigs, paths, node_count = get_the_segment_contig_map(contigfilepath)
      source_list, destination_list, weight_list = generate_edge_tensor(gfafilepath, segment_contigs)
      # try:
      feature_df, find_count = preprocess(csvfiles + "/" + datafolder + ".fasta.csv", truthfilepath)
      # feature_df = preprocess(csvfiles + "/" + datafolder + "_predictions.csv")
      # except:
      #   continue

      node_features = torch.DoubleTensor(feature_df[["Number of hits","Maximum coverage","Median coverage","Average coverage","Variance of coverage"]].values)
      edge_index = torch.tensor([source_list, destination_list], dtype=torch.long)
      edge_attr = torch.DoubleTensor(weight_list)
      y = torch.LongTensor(feature_df["Prediction"].values)

      train_set, validate_set, test_set = get_train_validate_test_data(0.8, 0.1, feature_df["train_set"].values)

      train_set = torch.BoolTensor(train_set)
      validate_set = torch.BoolTensor(validate_set)
      test_set = torch.BoolTensor(test_set)

      find_set = torch.BoolTensor(list(feature_df["find_set"].values))

      # Create data object
      data = Data(
          x=node_features, 
          edge_index=edge_index,
          edge_attr=edge_attr,
          y=y) 

      if _DEBUG:
        print("Prepare graphs using data....")

      G = to_networkx(data, to_undirected=True)

      if _DEBUG:
        print("Prepare visualize_graph using data....")

      # visualize_graph(G, color=data.y, filename = "Figures/" + datafolder + "_visualize_graph.png")

      if _DEBUG:
        print("Get the model ....")
      model = get_model(inputfeatures = data.x.shape[1], hidden_channels = 24, num_classes = 2)
      # print(model)

      if _DEBUG:
        print("eval_results of the model ....")
      eval_results = model.eval()
      # print(eval_results)

      if _DEBUG:
        print("Initial Test of the model ....")
      out, h = model(data.x, data.edge_index)
      if _DEBUG:
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

      iterate(model, optimizer, criterion, data, train_set, validate_set)

      if _DEBUG:
        print("Start the testinging phase....")

      test_acc, h, pred = test(model, data, test_set)

      model.eval()

      out, h = model(data.x, data.edge_index)
      # visualize(h, color=data.y, filename = "Figures/" + datafolder + "_results_visualize_embedding.png")

      if _DEBUG:
        return data, train_set, test_data
        break
      # break
