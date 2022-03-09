from process import process
from constants import csv_column_names, feature_column_names, pred_column_name

def main():
    userfolder = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset/10c25p"
    kmerfilepath = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4/10c25p-4mer"
    initpredictions = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults/10c25p.fasta.probs.out"

    gfaprefix = "assembly_graph_with_scaffolds.gfa"
    pathfilesuffix = "contigs.paths"
    fastafilesuffix = "contigs.fasta"

    fileseparator = "/"

    isKmerAvailable = True

    initial_predictions_csv_column_names = csv_column_names
    initial_predictions_feature_column_names = feature_column_names
    initial_predictions_pred_column_name = pred_column_name

    process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable, initial_predictions_csv_column_names, initial_predictions_feature_column_names, initial_predictions_pred_column_name)

main()