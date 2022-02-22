import pandas as pd

csv_column_names = ["ID", "Probability"]
upper_bound = 0.8
lower_bound = 0.2
fragment_limit = 10
pred_column_name = "Probability"

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"
def getClass(row):
    if float(row[pred_column_name]) < lower_bound:
        return 0
    elif float(row[pred_column_name]) > upper_bound:
        return 1
    else:
        return 0.5

def preprocess(filename):
    print("Filename:- ", filename)
    df = pd.read_csv(filename, sep = "\t", names = csv_column_names)
    df = df[csv_column_names]
    # df.to_csv(filename)
    df["Prediction"] = df.apply(lambda row: getClass(row), axis = 1)
    df["Prediction"] = df["Prediction"].astype(int)
    df["train_set"] = df.apply(lambda row: (not((lower_bound < row[pred_column_name] < upper_bound))), axis = 1)
    df["find_set"] = df.apply(lambda row: ((lower_bound < row[pred_column_name] < upper_bound)), axis = 1)
    return df, df["find_set"].sum()