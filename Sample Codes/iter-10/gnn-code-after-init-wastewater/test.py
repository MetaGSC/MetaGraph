from process import process
import os
import numpy as np
import pandas as pd

from math import exp, log

from constants import csv_column_names, feature_column_names, pred_column_name

gfaprefix = "assembly_graph_with_scaffolds.gfa"
pathfilesuffix = "contigs.paths"
fastafilesuffix = "contigs.fasta"

fileseparator = "/"

gnndatasetpath = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset"
csvfiles = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults"
kmer4folder = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4"
resultpath = "/home/hp/Acedemic/FYP/GNN/gnn-code/Results/withkmer"
resultwithoutkmerpath = "/home/hp/Acedemic/FYP/GNN/gnn-code/Results/withoutkmer"

def test():
    s = ""
    initial_predictions_csv_column_names = csv_column_names
    initial_predictions_feature_column_names = feature_column_names
    initial_predictions_pred_column_name = pred_column_name

    result_df = pd.DataFrame(columns = ["Name", "iskmer", "precision", "recall", "F1"])
    for datafolder in os.listdir(gnndatasetpath):
        userfolder = os.path.join(gnndatasetpath, datafolder)
        userfolder = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset/" + datafolder
        kmerfilepath = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4/" + datafolder + "-4mer"
        initpredictions = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults/" + datafolder + ".fasta.probs.out"

        isKmerAvailable = True

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # try:
        #     results, indexes = process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable, initial_predictions_csv_column_names, initial_predictions_feature_column_names, initial_predictions_pred_column_name, result_df)
        #     with open(resultpath + fileseparator + datafolder, "w") as file:
        #         probs = ""
        #         for row in results:
        #             probs += str(exp(row[1])/ (exp(row[1]) + exp(row[0]))) + ","
        #         probs = probs.strip(",")
        #         file.write(probs)
        #     with open(resultpath + "index" + fileseparator + datafolder, "w") as file:
        #         inds = ""
        #         for row in indexes:
        #             inds += str(row) + ","
        #         inds = inds.strip(",")
        #         file.write(inds)
        # except Exception as e:
        #     s+= userfolder + " :- " + str(e) + "\n"

        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        isKmerAvailable = False
        # try:
        results, indexes = process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable, initial_predictions_csv_column_names, initial_predictions_feature_column_names, initial_predictions_pred_column_name, result_df)
        with open(resultwithoutkmerpath + fileseparator + datafolder, "w") as file:
            probs = ""
            for row in results:
                # print("ROW", exp(row[1]), exp(row[1]), exp(row[0]))
                if (exp(row[1]) <= 0 and exp(row[0]) <= 0):
                    probs += '0.0,'
                else:
                    probs += str(exp(row[1])/ (exp(row[1]) + exp(row[0]))) + ","
            probs = probs.strip(",")
            file.write(probs)
        with open(resultwithoutkmerpath + "index" + fileseparator + datafolder, "w") as file:
            inds = ""
            for row in indexes:
                inds += str(row) + ","
            inds = inds.strip(",")
            file.write(inds)
                    
        # except Exception as e:
        #     s+= userfolder + " :- " + str(e) + "\n"

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # break
    print(s)
    result_df.to_csv("results_1.csv")
test()