import pandas as pd
import numpy as np
import re

csv_column_names = ["ID", "Number of hits","Maximum coverage","Median coverage","Average coverage","Variance of coverage","Best hit"]
upper_bound = 0.7
lower_bound = 0.4
fragment_limit = 10

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"
def label(row):
    if (row["Prediction"] == "plasmid"):
        return 1
    elif (row["Prediction"] == "chromosome"):
        return 0
    else:
        return 0.5

def preprocess(filename, truthfilepath):
    print("Filename:- ", filename)
    df = pd.read_csv(filename)
    df = df[csv_column_names]
    size = len(df)
    # df.to_csv(filename)
    df["Prediction"] = get_predictions_from_truth_file(truthfilepath, size)
    df["Prediction"] = df.apply(lambda row: label(row), axis = 1)
    # for i in df["Prediction"]:
    #     print(i)
    df["train_set"] = df["Prediction"] != 0.5
    df["find_set"] = df["Prediction"] == 0.5
    return df, df["find_set"].sum()

def get_predictions_from_truth_file(truthfilepath, size):
    predictions = []
    line_num = 0
    start = 'NODE_'
    end = '_length_'
    with open(truthfilepath) as file:
        line = file.readline()
        while (line != ""):
            line_num += 1
            line = line.split()
            contig_num = int((line[0].split("_")[1]).strip())
            # print(contig_num)
            if (str(line_num) != contig_num):
                for _ in range(line_num, contig_num):
                    line_num += 1
                    predictions.append(None)
            type_class = line[1].strip()
            if (type_class == "unclassified" or (not(type_class == "plasmid" or type_class == "chromosome"))):
                predictions.append(None)
            else:
                predictions.append(type_class)
            line = file.readline()
    if (size != contig_num):
        for _ in range(contig_num + 1, size + 1):
            predictions.append(None)
    return np.array(predictions)
    