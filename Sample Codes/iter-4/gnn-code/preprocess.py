import pandas as pd

csv_column_names = ["ID", "Number of hits","Maximum coverage","Median coverage","Average coverage","Variance of coverage","Best hit","Prediction"]
upper_bound = 0.8
lower_bound = 0.3
fragment_limit = 10

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"

def preprocess(filename):
    print("Filename:- ", filename)
    df = pd.read_csv(filename)
    df = df[csv_column_names]
    # df.to_csv(filename)
    df["Prediction"] = df["Prediction"] == "plasmid"
    df["Prediction"] = df["Prediction"].astype(int)
    df["train_set"] = df.apply(lambda row: (not((lower_bound < row["Average coverage"] < upper_bound) or row["Number of hits"] < fragment_limit)), axis = 1)
    df["find_set"] = df.apply(lambda row: ((lower_bound < row["Average coverage"] < upper_bound) or row["Number of hits"] < fragment_limit), axis = 1)
    return df