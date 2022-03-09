import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from prepare import get_the_segment_contig_map, generate_edge_tensor
from visualization import visualize_graph, visualize_embedding, visualize
from gnn_model import GCN, get_parameters, get_model, iterate, get_train_validate_test_data
from preprocess import preprocess, getGroundTruth, updatePreds, getPrecisionRecall

from constants import prediction_result_column, train_set_column, find_set_column, final_prdiction_probability, binary_prediction_column, feature_column_names

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

_DEBUG = False

def process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable):
  data = None
  all_csv_column_names = feature_column_names[:]

  gfafilepath = str(userfolder) + fileseparator + gfaprefix
  contigfilepath = str(userfolder) + fileseparator + pathfilesuffix
  fastafilepath = str(userfolder) + fileseparator + fastafilesuffix
  trutfilepath = str(userfolder) + fileseparator + "/truth"

  segment_contigs, paths, node_count = get_the_segment_contig_map(contigfilepath)
  source_list, destination_list, weight_list = generate_edge_tensor(gfafilepath, segment_contigs)
  # try:
  feature_df, find_count = preprocess(initpredictions, fastafilepath, kmerfilepath, isKmerAvailable, trutfilepath)
  # except:
  #   # continue
  #   print("Error Occured in Userfile ", userfolder)

  if isKmerAvailable:
    all_csv_column_names.extend([i for i in range(136)])
  node_features = torch.DoubleTensor(feature_df[all_csv_column_names].values)
  edge_index = torch.tensor([source_list, destination_list], dtype=torch.long)
  edge_attr = torch.DoubleTensor(weight_list)
  # gtruth_set = torch.LongTensor(list(feature_df["ground_truth"].values))

  y = torch.LongTensor(feature_df[prediction_result_column].values)

  train_set = torch.BoolTensor(feature_df[train_set_column].values)

  find_set = torch.BoolTensor(list(feature_df[find_set_column].values))
  binary_predictions_set = torch.LongTensor(list(feature_df[binary_prediction_column].values))

  # unclassified_ground_truth_set = torch.BoolTensor(list(feature_df["unclassified_truth"].values))

  ground_df = getGroundTruth(trutfilepath)

  # Create data object
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

  # visualize_graph(G, color=data.y, filename = "Figures/" + datafolder + "_visualize_graph.png")

  if _DEBUG:
    print("Get the model ....")
  model = get_model(inputfeatures = data.x.shape[1], hidden_channels = [24, 20, 10], num_classes = 2)
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

  iterate(model, optimizer, criterion, data, train_set)

  if _DEBUG:
    print("Start the testinging phase....")

  model.eval()

  out, h = model(data.x, data.edge_index)
  # visualize(h, color=data.y, filename = "Figures/" + datafolder + "_results_visualize_embedding.png")

  pred = out.argmax(dim=1)
  # test_correct = pred[find_set] == Binary_Prediction.values[find_set]
  
  # cfm = confusion_matrix(binary_predictions_set[find_set], pred[find_set], labels=[0,1])
  # cfm = confusion_matrix(gtruth_set[find_set], pred[find_set], labels=[0,1])

  if not(isKmerAvailable):
    getPrecisionRecall(ground_df["node_id"], ground_df["class"], find_set, binary_predictions_set, True)
  
  valid_pred = updatePreds(pred, binary_predictions_set, find_set)

  getPrecisionRecall(ground_df["node_id"], ground_df["class"], find_set, valid_pred, False)

  # print(valid_pred, "\n", pred , "\n", gtruth_set)
  # cfm = confusion_matrix(gtruth_set[find_set_column], valid_pred[find_set_column], labels=[0,1])
  # tn, fp, fn, tp = cfm.ravel()
  # precision = tp/ (tp+fp)
  # recall = tp/ (tp+fn)
  # f1 = 2*(precision*recall)/(precision+recall)

  # if isKmerAvailable:
  #   print(filename + ":- Withkmer")


  if _DEBUG:
    return data, train_set, test_data
    # break
  # break
  # feature_df["final_prdiction_probability"] = out.detach().numpy()
  # feature_df[id_column, final_prdiction_probability]
  return out.detach().numpy(), feature_df.index.values[find_set]
