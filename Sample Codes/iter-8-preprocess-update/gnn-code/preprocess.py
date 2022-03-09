import pandas as pd
from prepare import get_fasta_sequence_lengths

from constants import csv_column_names, pred_column_name, prediction_result_column, train_set_column, find_set_column, binary_prediction_column, id_column, length_column

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

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

def getTruthClass(row):
    if row["ground_truth"] == "plasmid":
        return 1
    elif row["ground_truth"] == "chromosome":
        return 0
    else:
        return None

def getNodeId(row):
    name = row["name"]
    return int(name[5:])

# def adjustTrainClass(row):
#     if row["ground_truth"] == 0:
#         return row[train_set_column]
#     elif row["ground_truth"] == 1:
#         return row[train_set_column]
#     else:
#         return False

# def adjustFindClass(row):
#     if row["ground_truth"] == 0:
#         return row[find_set_column]
#     elif row["ground_truth"] == 1:
#         return row[find_set_column]
#     else:
#         return True

# def adjustUtruthClass(row):
#     if row["ground_truth"] == 0:
#         return True
#     elif row["ground_truth"] == 1:
#         return True
#     else:
#         return False

def preprocess(filename, fastafilepath, kmerfilepath, isKmerAvailable, trutfilepath):
    print("Filename:- ", filename)
    all_csv_column_names = csv_column_names[:]
    df = pd.read_csv(filename, sep = "\t", names = csv_column_names)

    length_map = get_fasta_sequence_lengths(fastafilepath)
    df[length_column] = df.apply(lambda row: length_map[row[id_column]], axis = 1)

    if isKmerAvailable:
        kmer_df = pd.read_csv(kmerfilepath, sep = " ", names = [i for i in range(136)])
        df = pd.concat([df, kmer_df], axis = 1)
        all_csv_column_names.extend([i for i in range(136)])
    df = df[all_csv_column_names]

    df[prediction_result_column] = df.apply(lambda row: getClass(row), axis = 1)
    df[prediction_result_column] = df[prediction_result_column].astype(int)

    df[binary_prediction_column] = df.apply(lambda row: getPred(row), axis = 1)


    df[train_set_column] = df.apply(lambda row: (not((lower_bound < row[pred_column_name] < upper_bound))), axis = 1)
    df[find_set_column] = df.apply(lambda row: ((lower_bound < row[pred_column_name] < upper_bound)), axis = 1)

    # df["unclassified_truth"] = False

    # df[find_set_column] = df.apply(lambda row: adjustFindClass(row), axis = 1)
    # df[train_set_column] = df.apply(lambda row: adjustTrainClass(row), axis = 1)
    # df["unclassified_truth"] = df.apply(lambda row: adjustUtruthClass(row), axis = 1)

    # df[["unclassified_truth"]].to_csv("df.csv")

    return df, df[find_set_column].sum()

def getGroundTruth(trutfilepath):
    g_df = pd.read_csv(trutfilepath, sep = "\t", names = ["name", "ground_truth"])
    g_df["class"] = g_df.apply(lambda row: getTruthClass(row), axis = 1)
    g_df["node_id"] = g_df.apply(lambda row: getNodeId(row), axis = 1)
    # Remove class = None
    g_df = g_df.dropna()
    return g_df

def updatePreds(pred, binary_predictions_set, find_set):
    valid_pred = binary_predictions_set
    data_index = -1
    for element in find_set:
        data_index += 1
        if element:
            valid_pred[data_index] = pred[data_index]
    return valid_pred

def getPrecisionRecall(nodeids, classes, find_set, pred_set, isTool):
    if isTool:
        toolName = "Plass Class"
    else:
        toolName = "MetaPC"    

    results, truthresults = generateTruthResults(nodeids, classes, find_set, pred_set)

    cfm = confusion_matrix(results, truthresults, labels=[0,1])
    tn, fp, fn, tp = cfm.ravel()

    precision = ((tp/ (tp+fp)) + (tn/ (tn+fn))) / 2
    recall = ((tp/ (tp+fn)) + (tn/ (tn+fp))) / 2

    f1 = 2*(precision*recall)/(precision+recall)

    print("\n===" + toolName + " Results===\n", cfm, "\nprecision", precision, "recall", recall, "f1", f1, "\n===" + toolName + "Results===\n")

# def getPrecisionRecallForTool(results, truthresults):
#     cfm = confusion_matrix(results, truthresults, labels=[0,1])
#     tn, fp, fn, tp = cfm.ravel()

#     precision = ((tp/ (tp+fp)) + (tn/ (tn+fn))) / 2
#     recall = ((tp/ (tp+fn)) + (tn/ (tn+fp))) / 2

#     f1 = 2*(precision*recall)/(precision+recall)

#     print("\n===PlassClass Results===\n", cfm, "\nprecision", precision, "recall", recall, "f1", f1, "\n===PlassClass Results===\n")


def generateTruthResults(nodeids, classes, find_set, pred_set):
    pred_array = []
    class_array = []
    for index in range(len(find_set)):
        if index in nodeids:
            boolean_value = find_set[index]
            if (boolean_value):
                pred_array.append(pred_set[index])
                class_array.append(classes[index])
    return pred_array, class_array
