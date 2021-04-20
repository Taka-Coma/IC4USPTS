# -*- coding: utf-8 -*-

import numpy as np

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
import xgboost as xgb

### Evaluation
from ..utils.evaluation import calc_scores


def main():
    vec_path = './vectors/setencesTrans_paraphrase.dump'

    with open(vec_path, 'rb') as r:
        dataset = pickle.load(r) 

	### Examining i-th data in leave-one-out manner
    for i in range(len(dataset)):
        X_test, y_test = dataset[i]
        y_test = [0 if v == '-1' else int(v) for v in y_test]
        X_train, y_train = genTrain(dataset, i)

        dtrain = xgb.DMatrix(X_train.tocsr(), label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
            }

        model = xgb.train(params, dtrain, 20)
        predicts = np.round(model.predict(dtest))

        accuracy, precision, recall, f1, f2, tpr, tnr, gmean = calc_scores(y_test, predicts)
        scores = {'gmean': gmean, 'tpr': tpr, 'tnr': tnr, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2}



def genTrain(dataset, except_ind):
    y = []
    X = None
    for i in range(len(dataset)):
        if i == except_ind:
            continue

        if X is None:
            X = dataset[i][0].copy()
        else:
            X = vstack((X, dataset[i][0]))

        y.extend(dataset[i][1])

    y_out = [0 if v == '-1' else int(v) for v in y]
    return X, y_out



if __name__ == "__main__":
    main()
