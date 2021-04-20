import numpy as np
from scipy.stats import norm

from collections import Counter

from sklearn.base import clone

from metric_learn import LMNN
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split as split


class MMEnsembleClassifier():
    def __init__(self,
        sampling_range=list(np.arange(0.2, 2.1, 0.2)),
        weight_strategy='Auto',
        mu=1.0,
        sigma=0.2,
        base_estimator=None,
        a=1,
        k=1,
        metric_learner=LMNN(),
        n_jobs=1
    ):
        self.base_estimator = base_estimator
        self.sampling_range = sampling_range
        self.weight_strategy = weight_strategy
        self.a = a
        self.k = k
        self.metric_learner = metric_learner
        self.n_jobs = n_jobs

        if weight_strategy == 'Gauss':
            sampling_weights = [norm.pdf(x, loc=mu, scale=sigma) for x in self.sampling_range]
        else:
            sampling_weights = [1]*len(sampling_range)
        self.sampling_weights = sampling_weights


    def fit(self, X, y):
        self.estimators_ = []

        if self.weight_strategy == 'Uniform' or self.weight_strategy == 'Gauss':
            self._fit(X, y)
        else:
            X_train, X_valid, y_train, y_valid = split(X, y, test_size=0.1, stratify=y)

            counter = Counter(y_train)
            freq = counter.most_common(2)
            minor_count = freq[1][1]

            corrects = []
            for rate in self.sampling_range:
                if self.base_estimator is None:
                    estimator = MLEnsembleClassifier(sampling_strategy={
                        freq[0][0]: int(minor_count*rate),
                        freq[1][0]: minor_count
                    }, metric_learner=self.metric_learner, n_jobs=self.n_jobs)
                else:
                    estimator = MLEnsembleClassifier(sampling_strategy={
                        freq[0][0]: int(minor_count*rate),
                        freq[1][0]: minor_count
                    }, metric_learner=self.metric_learner, 
                    base_estimator=self.base_estimator, n_jobs=self.n_jobs)
                try:
                    estimator.fit(X, y)
                except ValueError as err:
                    corrects.append([])
                    print(f'ValueError: {err} at {rate}')
                    continue
                self.estimators_.append(estimator)
                self.classes_ = estimator.classes_

                y_pred = estimator.predict(X_valid)
                corrects.append([i for i, x in enumerate(zip(y_valid, y_pred)) if x[0] == x[1] ])

            counter = Counter()
            for correct in corrects:
                c = Counter(correct)
                counter += c

            counter = dict(counter)

            for i, correct in enumerate(corrects):
                if len(correct) == 0:
                    self.sampling_weights[i] = [0]
                else:
                    weight = [counter[ind] for ind in correct]
                    self.sampling_weights[i] = weight
            self._fit(X, y)

        return self 


    def _fit(self, X, y):
        counter = Counter(y)
        freq = counter.most_common(2)
        minor_count = freq[1][1]

        for rate in self.sampling_range:
            if self.base_estimator is None:
                estimator = MLEnsembleClassifier(sampling_strategy={
                    freq[0][0]: int(minor_count*rate),
                    freq[1][0]: minor_count
                }, metric_learner=self.metric_learner, n_jobs=self.n_jobs)
            else:
                estimator = MLEnsembleClassifier(sampling_strategy={
                    freq[0][0]: int(minor_count*rate),
                    freq[1][0]: minor_count
                }, metric_learner=self.metric_learner,
                base_estimator=self.base_estimator, n_jobs=self.n_jobs)
            try:
                estimator.fit(X, y)
            except ValueError as err:
                print(f'ValueError: {err} at {rate}')
                continue
            self.estimators_.append(estimator)
            self.classes_ = estimator.classes_
        return self


    def set_params(self, a=1, k=1):
        self.a = a
        self.k = k


    def predict_proba(self, X):
        probas = 0
        for estimator, weight in zip(self.estimators_, self.sampling_weights):
            if self.weight_strategy == 'Auto':
                w = sum([0 if ww == 0 else  self.a*ww**(-self.k) for ww in weight])
                probas += estimator.predict_proba(X)*w
            else:
                probas += estimator.predict_proba(X)*weight
        psum = np.sum(probas, axis=1)
        probas = probas / np.transpose(np.array([psum, psum]))
        return probas


    def predict(self, X):
        proba = self.predict_proba(X)
        pred = self.classes_.take((np.argmax(proba, axis=1)), axis=0)
        return pred


class MLEnsembleClassifier(BalancedBaggingClassifier):
    def __init__(self,
        sampling_strategy='auto',
        metric_learner=LMNN(),
        base_estimator=AdaBoostClassifier(),
        n_jobs=1,
        n_estimators=10
        ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
			replacement=True,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            n_jobs=n_jobs,
            sampling_strategy=sampling_strategy,
        )
        self.base_estimator = Pipeline([
            ('metric learner', clone(metric_learner)),
            ('classifier', clone(base_estimator))
        ])
