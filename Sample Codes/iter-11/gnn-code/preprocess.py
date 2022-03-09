import pandas as pd
from prepare import get_fasta_sequence_lengths

from constants import PLASMID, CHROMOSOME, prediction_result_column, train_set_column, find_set_column, binary_prediction_column, id_column, length_column, ground_truth_column, name_column, class_column, node_id_column

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# csv_column_names = ["ID", "Probability", "length"]
upper_bound = 0.8
lower_bound = 0.2
fragment_limit = 10
# pred_column_name = "Probability"

# root = "/home/hp/FYP/GNN/gnn-code/rawdata/raw"
def getClass(row, pred_column_name):
    if float(row[pred_column_name]) < lower_bound:
        return 0
    elif float(row[pred_column_name]) > upper_bound:
        return 1
    else:
        return row[pred_column_name]

def getPred(row, pred_column_name):
    if float(row[pred_column_name]) <= 0.5:
        return 0
    else:
        return 1

def getTruthClass(row):
    if row[ground_truth_column] == PLASMID:
        return 1
    elif row[ground_truth_column] == CHROMOSOME:
        return 0
    else:
        return None

def getNodeId(row):
    name = row[name_column]
    return int(name[5:])

def preprocess(filename, fastafilepath, kmerfilepath, isKmerAvailable, trutfilepath, csv_column_names, pred_column_name):
    print("Filename:- ", filename)
    all_csv_column_names = csv_column_names[:]
    df = pd.read_csv(filename, sep = "\t", names = csv_column_names)
    print(df.dtypes)

    length_map = get_fasta_sequence_lengths(fastafilepath)
    # df[length_column] = df.apply(lambda row: length_map[row[id_column]], axis = 1)

    if isKmerAvailable:
        kmer_df = pd.read_csv(kmerfilepath, sep = " ", names = [i for i in range(136)])
        df = pd.concat([df, kmer_df], axis = 1)
        all_csv_column_names.extend([i for i in range(136)])
    df = df[all_csv_column_names]

    df[prediction_result_column] = df.apply(lambda row: getClass(row, pred_column_name), axis = 1)
    df[prediction_result_column] = df[prediction_result_column].astype(float)

    df[binary_prediction_column] = df.apply(lambda row: getPred(row, pred_column_name), axis = 1)


    df[train_set_column] = df.apply(lambda row: (not((lower_bound < row[pred_column_name] < upper_bound))), axis = 1)
    df[find_set_column] = df.apply(lambda row: ((lower_bound < row[pred_column_name] < upper_bound)), axis = 1)

    # df["unclassified_truth"] = False

    # df[find_set_column] = df.apply(lambda row: adjustFindClass(row), axis = 1)
    # df[train_set_column] = df.apply(lambda row: adjustTrainClass(row), axis = 1)
    # df["unclassified_truth"] = df.apply(lambda row: adjustUtruthClass(row), axis = 1)

    # df[["unclassified_truth"]].to_csv("df.csv")

    return df, df[find_set_column].sum()

def getGroundTruth(trutfilepath):
    g_df = pd.read_csv(trutfilepath, sep = "\t", names = [name_column, ground_truth_column])
    g_df[class_column] = g_df.apply(lambda row: getTruthClass(row), axis = 1)
    g_df[node_id_column] = g_df.apply(lambda row: getNodeId(row), axis = 1)
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

def getPrecisionRecall(nodeids, classes, find_set, pred_set, isTool, userfolder, result_df, isKmerAvailable):
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

    name = userfolder.split("/")[-1]
    result_df.loc[len(result_df)] =[name, isKmerAvailable, precision, recall, f1]

    print("\n===" + toolName + " Results===\n", cfm, "\nprecision", precision, "recall", recall, "f1", f1, "\n===" + toolName + "Results===\n")

    return result_df

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
