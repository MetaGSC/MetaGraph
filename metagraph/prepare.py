import re

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from pysam import FastaFile

from predict import predict
from Bio import SeqIO

_DEBUG = False

def get_fasta_sequence_lengths(fastafile):
  length_map = {}
  for record in SeqIO.parse(fastafile, 'fasta'):
      length_map[record.id] = len(record.seq)
  return length_map

def get_label_sequences(fastafile):
  fasta_object = FastaFile(fastafile)
  fasta_references = fasta_object.references

  for name in fasta_references:
    attributes = {}

    contig_num = int(str(int(re.search('%s(.*)%s' % (start, end), name).group(1))-1))
    predicted_values = predict(fasta_object.fetch(reference = name))

    attributes["features"] = predicted_values
    attributes["class"] = 0.5
    attributes["train_sem"] = False

    # TODO Change this
    color_value = sum(predicted_values) / len(predicted_values)

    if color_value < 0.4:
      attributes["class"] = 0
      attributes["train_sem"] = True
    elif color_value > 0.8:
      attributes["class"] = 1
      attributes["train_sem"] = True

    features[contig_num] = attributes
  return features

def get_the_segment_contig_map(contigfilepath):
  paths = {}
  segment_contigs = {}
  node_count = 0

  with open(contigfilepath) as file:
    name = file.readline()
    path = file.readline()
    
    while name != "" and path != "":
            
        while ";" in path:
            path = path[:-2]+","+file.readline()
        
        start = 'NODE_'
        end = '_length_'
        contig_num = int(str(int(re.search('%s(.*)%s' % (start, end), name).group(1))-1))
        
        segments = path.rstrip().split(",")
        
        if contig_num not in paths:
            node_count += 1
            paths[contig_num] = [segments[0], segments[-1]]
        
        for segment in segments:
            # TODO Check this
            if segment.endswith("+") or segment.endswith("-"):
                segment = segment[:-1]
            if segment not in segment_contigs:
                segment_contigs[segment] = set([contig_num])
            else:
                segment_contigs[segment].add(contig_num)
        
        name = file.readline()
        path = file.readline()

    return segment_contigs, paths, node_count

def adjust_weights(weight_list):
  summation = 0
  count = 0
  none_indexes = []
  for i in range(len(weight_list)):
    if weight_list[i] == None:
      none_indexes.append(i)
      continue
    count += 1
    summation += weight_list[i][0]
  
  average_weight = summation / count
  for i in none_indexes:
    weight_list[i] = [average_weight]

  return weight_list

#make segment_contigs global
def generate_edge_tensor(gfafilepath, segment_contigs):
  source_list = []
  destination_list= []
  weight_list = []
  isNeedToAdjustWeights = False

  with open(gfafilepath) as file:
    line = file.readline()
    if _DEBUG:
      line_count = 1
    while line != "":
      # Identify lines with link information
      if "L" in line:
          strings = line.split("\t")
          seg1, seg2 = strings[1], strings[3]
          try:
            weight = strings[5].strip()
          except:
            print(line_count, "Links between Contigs nedd to be separated by a tab in .gfa file!")
          if seg1 not in segment_contigs or seg2 not in segment_contigs:
            line = file.readline()
            if _DEBUG:
              line_count += 1
            continue
          contig1 = segment_contigs[seg1]
          contig2 = segment_contigs[seg2]
          if _DEBUG:
            print(line_count, contig1, contig2)
          for cont1 in contig1:
            for cont2 in contig2:
              if cont1 != cont2:
                source_list.append(cont1)
                destination_list.append(cont2)
                if weight.isnumeric():
                  weight_list.append([int(weight)])
                elif weight[:-1].isnumeric():
                  weight_list.append([int(weight[:-1])])
                else:
                  weight_list.append(None)
                  isNeedToAdjustWeights = True
      line = file.readline()
      if _DEBUG:
        line_count += 1
        # print(line_count)
    if _DEBUG:
      print("Finished the reading part of GFA File")
    if isNeedToAdjustWeights:
      weight_list = adjust_weights(weight_list)
    return source_list, destination_list, weight_list
