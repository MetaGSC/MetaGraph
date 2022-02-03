csv_column_names = ["seq_id", "fragment_count","kmer_plas_prob","biomer_plas_prob", "final_plas_prob", "class"]
upper_bound = 0.8
lower_bound = 0.3
fragment_limit = 10

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"

def preprocess(filename):
    df = pd.read_csv(filename)
    df = df[csv_column_names]
    # df.to_csv(filename)
    df["class"] = df["class"] == "plasmid"
    df["class"] = df["class"].astype(int)
    df["train_set"] = df.apply(lambda row: (not((lower_bound < row["final_plas_prob"] < upper_bound) or row["fragment_count"] < fragment_limit)), axis = 1)
    df["find_set"] = df.apply(lambda row: ((lower_bound < row["final_plas_prob"] < upper_bound) or row["fragment_count"] < fragment_limit), axis = 1)
    return df