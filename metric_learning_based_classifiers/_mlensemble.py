import numpy as np
from scipy.stats import norm

from collections import Counter

from sklearn.base import clone

from metric_learn import LMNN
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split as split


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
