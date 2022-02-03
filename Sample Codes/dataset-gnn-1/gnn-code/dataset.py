import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import os

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class UserDataset(Dataset):
    def __init__(self, root, raw_file, path, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        # self.filename = filename
        self.raw_file = raw_file
        self.path = path
        self.src = root
        # self.processed_dir = processed_dir
        self.train_set = None
        self.cv_set = None
        self.test_set = None
        self.find_set = None
        self.data = data
        super(UserDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.raw_file

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = torch.load(os.path.join(self.processed_dir, f'data_test_{(self.get_name(self.raw_paths[0]))}.pt'))

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def get_name(self, file):
        name =  os.path.splitext(file)[0]
        if "/" not in name:
          return name
        else:
          return (name.split("/"))[-1]

    def process(self):
        gfafilepath = self.path + "/assembly_graph_with_scaffolds.gfa"
        contigfilepath = self.path + "/contigs.paths"
        fastafilepath = self.path + "/contigs.fasta"

        segment_contigs, paths, node_count = get_the_segment_contig_map(contigfilepath)
        source_list, destination_list, weight_list = generate_edge_tensor(gfafilepath, segment_contigs)
        feature_df = preprocess(self.raw_paths[0])

        node_features = torch.DoubleTensor(feature_df[["fragment_count","kmer_plas_prob","biomer_plas_prob", "final_plas_prob"]].values)
        edge_index = torch.tensor([source_list, destination_list], dtype=torch.long)
        edge_attr = torch.DoubleTensor(weight_list)
        y = feature_df["class"]

        self.train_set = feature_df["train_set"].values
        self.find_set = feature_df["find_set"].values

        # Create data object
        data = Data(
            x=node_features, 
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y) 
        
        if self.test:
            torch.save(data, 
                os.path.join(self.processed_dir, 
                              f'data_test_{(self.get_name(self.raw_paths[0]))}.pt'))
        else:
            torch.save(data, 
                os.path.join(self.processed_dir, 
                              f'data_{(self.get_name(self.raw_paths[0]))}.pt'))
            
        # data, slices = self.collate([self.data])
        # torch.save((data, slices), self.processed_dir)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{(self.get_name(self.raw_paths[0]))}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{(self.get_name(self.raw_paths[0]))}.pt'))   
        return data