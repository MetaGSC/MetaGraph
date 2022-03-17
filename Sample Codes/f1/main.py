from process import process
from constants import csv_column_names, feature_column_names, pred_column_name
import pandas as pd

def main():
    userfolder = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset/2c5p_low_copy"
    kmerfilepath = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4/2c5p_low_copy-4mer"
    initpredictions = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults/2c5p_low_copy.fasta.probs.out"

    gfaprefix = "assembly_graph_with_scaffolds.gfa"
    pathfilesuffix = "contigs.paths"
    fastafilesuffix = "contigs.fasta"

    fileseparator = "/"

    isKmerAvailable = False

    initial_predictions_csv_column_names = csv_column_names
    initial_predictions_feature_column_names = feature_column_names
    initial_predictions_pred_column_name = pred_column_name

    result_df = pd.DataFrame(columns = ["Name", "iskmer", "precision", "recall", "F1"])

    process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable, initial_predictions_csv_column_names, initial_predictions_feature_column_names, initial_predictions_pred_column_name, result_df)

main()