from process import process

def main():
    userfolder = "/home/hp/Acedemic/FYP/GNN/Gnn-Dataset/10c25p"
    kmerfilepath = "/home/hp/Acedemic/FYP/GNN/gnn-code/kmer4/10c25p-4mer"
    initpredictions = "/home/hp/Acedemic/FYP/GNN/gnn-code/plasclassresults/10c25p.fasta.probs.out"

    gfaprefix = "assembly_graph_with_scaffolds.gfa"
    pathfilesuffix = "contigs.paths"
    fastafilesuffix = "contigs.fasta"

    fileseparator = "/"

    isKmerAvailable = True

    process(userfolder, initpredictions, kmerfilepath, gfaprefix, pathfilesuffix, fastafilesuffix, fileseparator, isKmerAvailable)

main()