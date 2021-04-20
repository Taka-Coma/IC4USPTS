# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import RandomForestClassifier as RF

### Evaluation
from ..utils.evaluation import calc_scores

def main():
	vec_path = './vectors/setencesTrans_paraphrase.dump'

    with open(vec_path, 'rb') as r:
        dataset = pickle.load(r) 

	classifier_list = ['knn', 'rf', 'svmlin', 'svmrbf','lr']
	cls_name = classifier_list[0]

    for i in range(len(dataset)):
        X_test, y_test = dataset[i]
        X_train, y_train = genTrain(dataset, i)

        if cls_name == 'rf':
            cls = RF(n_jobs=-1)
        elif cls_name == 'svmlin':
            cls = LinearSVC()
        elif cls_name == 'lr':
            cls = LR()
        elif cls_name == 'svmrbf':
            cls = SVC(kernel='rbf')
        elif cls_name == 'knn':
            cls = kNN(n_jobs=-1)

        cls.fit(X_train, y_train)
        predicts = cls.predict(X_test)

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
    return X, y



if __name__ == "__main__":
    main()
