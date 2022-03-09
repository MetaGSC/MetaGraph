from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


def getPlassclassData(results, truthresults):
    cfm = confusion_matrix(results, truthresults, labels=[0,1])
    tn, fp, fn, tp = cfm.ravel()

    precision = ((tp/ (tp+fp)) + (tn/ (tn+fn))) / 2
    recall = ((tp/ (tp+fn)) + (tn/ (tn+fp))) / 2

    f1 = 2*(precision*recall)/(precision+recall)

    print("\n===PlassClass Results===\n", cfm, "\nprecision", precision, "recall", recall, "f1", f1, "\n===PlassClass Results===\n")
