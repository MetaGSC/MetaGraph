import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import pandas as pd

from prepare import get_the_segment_contig_map, generate_edge_tensor
from visualization import visualize_graph, visualize_embedding, visualize
from gnn_model import GCN, get_parameters, get_model, iterate, get_train_validate_test_data
from preprocess import preprocess, getGroundTruth, updatePreds, getPrecisionRecall

from constants import class_column, node_id_column,prediction_result_column, train_set_column, find_set_column, final_prdiction_probability, binary_prediction_column

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

_DEBUG = False

def process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable, csv_column_names, feature_column_names, pred_column_name, result_df):
  data = None
  all_csv_column_names = feature_column_names[:]

  gfafilepath = str(userfolder) + fileseparator + gfaprefix
  contigfilepath = str(userfolder) + fileseparator + pathfilesuffix
  fastafilepath = str(userfolder) + fileseparator + fastafilesuffix
  trutfilepath = str(userfolder) + fileseparator + "/truth"

  segment_contigs, paths, node_count = get_the_segment_contig_map(contigfilepath)
  source_list, destination_list, weight_list = generate_edge_tensor(gfafilepath, segment_contigs)
  feature_df, find_count = preprocess(initpredictions, fastafilepath, kmerfilepath, isKmerAvailable, trutfilepath, csv_column_names, pred_column_name)

  if isKmerAvailable:
    all_csv_column_names.extend([i for i in range(136)])
  node_features = torch.DoubleTensor(feature_df[all_csv_column_names].values)
  edge_index = torch.tensor([source_list, destination_list], dtype=torch.long)
  edge_attr = torch.DoubleTensor(weight_list)

  y = torch.LongTensor(feature_df[prediction_result_column].values)

  train_set = torch.BoolTensor(feature_df[train_set_column].values)

  find_set = torch.BoolTensor(list(feature_df[find_set_column].values))
  binary_predictions_set = torch.LongTensor(list(feature_df[binary_prediction_column].values))


  ground_df = getGroundTruth(trutfilepath)

  data = Data(
      x=node_features, 
      edge_index=edge_index,
      edge_attr=edge_attr,
      y=y) 
  print("Shape:- ", data.x.shape)
  if _DEBUG:
    print("Prepare graphs using data....")

  G = to_networkx(data, to_undirected=True)

  if _DEBUG:
    print("Prepare visualize_graph using data....")

  if _DEBUG:
    print("Get the model ....")
  model = get_model(inputfeatures = data.x.shape[1], hidden_channels = [69, 20, 10], num_classes = 2)
  print(model)

  if _DEBUG:
    print("eval_results of the model ....")
  eval_results = model.eval()

  if _DEBUG:
    print("Initial Test of the model ....")
  out, h = model(data.x, data.edge_index)
  if _DEBUG:
    print(f'Embedding shape: {list(h.shape)}')

  if _DEBUG:
    print("Initial visualize_embedding of the model ....")

  if _DEBUG:
    print("Initial visualize of the model ....")


  optimizer, criterion = get_parameters(model)

  if _DEBUG:
    print("Start the training phase....")

  iterate(model, optimizer, criterion, data, train_set)

  if _DEBUG:
    print("Start the testinging phase....")

  model.eval()

  out, h = model(data.x, data.edge_index)

  pred = out.argmax(dim=1)

  if not(isKmerAvailable):
    result_df = getPrecisionRecall(ground_df[node_id_column], ground_df[class_column], find_set, binary_predictions_set, True, userfolder, result_df, isKmerAvailable)
  
  valid_pred = updatePreds(pred, binary_predictions_set, find_set)

  result_df = getPrecisionRecall(ground_df[node_id_column], ground_df[class_column], find_set, valid_pred, False, userfolder, result_df, isKmerAvailable)

  if _DEBUG:
    return data, train_set, test_data
  return out.detach().numpy(), feature_df.index.values[find_set]
