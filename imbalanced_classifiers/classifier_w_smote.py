# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN

### Preprocessing 
from smote_variants import SMOTE, ProWSyn, polynom_fit_SMOTE as PFSMOTE

### Evaluation
from ..utils.evaluation import calc_scores


def main():
    vec_path = './vectors/setencesTrans_paraphrase.dump'

    oss = ['SMOTE', 'ProWSyn', 'PFSMOTE']
	os_name = oss[0]

    with open(vec_path, 'rb') as r:
        dataset = pickle.load(r) 

    classifier_list = ['knn', 'rf', 'svmlin', 'svmrbf', 'lr']
	cls_name = classifier_list[0]

    for i in range(len(dataset)):
        X_test, y_test = dataset[i]
        y_test = [0 if v == '-1' else int(v) for v in y_test]
        X_train, y_train = genTrain(dataset, i)
        X_train = X_train.toarray()

		### Oversampling
        if os_name == 'SMOTE':
            sampler = SMOTE(n_jobs=-1)
        elif os_name == 'ProWSyn':
            sampler = ProWSyn(n_jobs=-1)
        elif os_name == 'PFSMOTE':
            sampler = PFSMOTE()

        X_train, y_train = sampler.sample(X_train, y_train)

        if cls_name == 'rf':
            cls = RF(n_jobs=-1)
        elif cls_name == 'lr':
            cls = LR()
        elif cls_name == 'svmlin':
            cls = LinearSVC()
        elif cls_name == 'svmrbf':
            cls = SVC(kernel='rbf')
        elif cls_name == 'knn':
            cls = KNN(n_jobs=-1)

        model = cls
        model.fit(X_train, y_train)
        predicts = model.predict(X_test)

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
