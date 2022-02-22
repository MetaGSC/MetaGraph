import pandas as pd
from prepare import get_fasta_sequence_lengths

from constants import csv_column_names, pred_column_name, prediction_result_column, train_set_column, find_set_column, binary_prediction_column, id_column, length_column

# csv_column_names = ["ID", "Probability", "length"]
upper_bound = 0.8
lower_bound = 0.2
fragment_limit = 10
# pred_column_name = "Probability"

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"
def getClass(row):
    if float(row[pred_column_name]) < lower_bound:
        return 0
    elif float(row[pred_column_name]) > upper_bound:
        return 1
    else:
        return row[pred_column_name]

def getPred(row):
    if float(row[pred_column_name]) <= 0.5:
        return 0
    else:
        return 1

def preprocess(filename, fastafilepath):
    print("Filename:- ", filename)
    df = pd.read_csv(filename, sep = "\t", names = csv_column_names)

    length_map = get_fasta_sequence_lengths(fastafilepath)
    df[length_column] = df.apply(lambda row: length_map[row[id_column]], axis = 1)
    
    df = df[csv_column_names]
    df[prediction_result_column] = df.apply(lambda row: getClass(row), axis = 1)
    df[binary_prediction_column] = df.apply(lambda row: getPred(row), axis = 1)
    df[prediction_result_column] = df[prediction_result_column].astype(int)
    df[train_set_column] = df.apply(lambda row: (not((lower_bound < row[pred_column_name] < upper_bound))), axis = 1)
    df[find_set_column] = df.apply(lambda row: ((lower_bound < row[pred_column_name] < upper_bound)), axis = 1)
    return df, df[find_set_column].sum()