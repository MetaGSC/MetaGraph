from process import process
import os
import numpy as np

from math import exp 

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
    for datafolder in os.listdir(gnndatasetpath):
        userfolder = os.path.join(gnndatasetpath, datafolder)
        userfolder = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset/" + datafolder
        kmerfilepath = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4/" + datafolder + "-4mer"
        initpredictions = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults/" + datafolder + ".fasta.probs.out"

        # isKmerAvailable = True

        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # # try:
        # results, indexes = process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable)
        # with open(resultpath + fileseparator + datafolder, "w") as file:
        #     probs = ""
        #     for row in results:
        #         probs += str(exp(row[1])/ (exp(row[1]) + exp(row[0]))) + ","
        #     probs = probs.strip(",")
        #     file.write(probs)
        # with open(resultpath + "index" + fileseparator + datafolder, "w") as file:
        #     inds = ""
        #     for row in indexes:
        #         inds += str(row) + ","
        #     inds = inds.strip(",")
        #     file.write(inds)
        # # except Exception as e:
        # #     s+= userfolder + " :- " + str(e) + "\n"

        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        isKmerAvailable = False
        try:
            results, indexes = process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable)
            with open(resultwithoutkmerpath + fileseparator + datafolder, "w") as file:
                probs = ""
                for row in results:
                    probs += str(exp(row[1])/ (exp(row[1]) + exp(row[0]))) + ","
                probs = probs.strip(",")
                file.write(probs)
            with open(resultwithoutkmerpath + "index" + fileseparator + datafolder, "w") as file:
                inds = ""
                for row in indexes:
                    inds += str(row) + ","
                inds = inds.strip(",")
                file.write(inds)
                    
        except Exception as e:
            s+= userfolder + " :- " + str(e) + "\n"

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # break
    print(s)
test()