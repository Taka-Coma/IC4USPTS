# -*- coding: utf-8 -*-

### Evaluation
from sklearn.metrics import confusion_matrix as cm
from math import sqrt

def calc_scores(y_true, y_pred):
    tn, fp, fn, tp = cm(y_true, y_pred).ravel()
    if tn + fp > fn + tp:
        tpr = float(tp)/(fn+tp)
        tnr = float(tn)/(tn+fp) 
    else:
        tnr = float(tp)/(fn+tp)
        tpr = float(tn)/(tn+fp) 

    ## precision, recall, fscore
    if tp+fp == 0:
        precision = 0
    else:
        precision = float(tp) / (tp+fp)
    recall = tpr
    f1 = harmonicMean(precision, recall)
    f2 = harmonicMean(precision, recall, beta=2)

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return accuracy, precision, recall, f1, f2, tpr, tnr, sqrt(tpr * tnr)


def harmonicMean(f1, f2, beta=1):
    if f1 == 0 and f2 == 0:
        return 0
    else:
        return (1+beta**2)*f1*f2 / ((beta**2) * f1 + f2)
