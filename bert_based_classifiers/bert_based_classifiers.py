#!/usr/bin/env python
# coding: utf-8

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd

from math import sqrt
from glob import glob
import numpy as np
import math

import multiprocessing

models = {
	'bert': 'bert-large-uncased', 
	'roberta': 'roberta-large', 
	'albert': 'albert-large-v2', 
	'deberta': 'microsoft/deberta-large', 
	'xlnet': 'xlnet-large-cased',
}

def main():
	model_names = ['bert', 'roberta', 'albert', 'deberta', 'xlnet']
	mname = model_names[0]

    lrs = [9e-6, 5e-6, 3e-6, 9e-5, 5e-5, 3e-5]
    lr = lrs[0]

    for i in range(50):
        p = multiprocessing.Process(target=test,
            args=(mname, i, lr))
        p.start()
        p.join()


def test(mname, i=0, init_lr=3e-5):
    epochs = 20

    ### Prepare datasets
    train_df, test_df, pos, neg = loadData(i)

    class_weight = [
        math.log((pos+neg)/neg),
        math.log((pos+neg)/pos)
    ]

    model_args = ClassificationArgs(
        do_lower_case=True,
        evaluate_during_training_verbose=True,
        num_train_epochs=epochs,
        overwrite_output_dir=True,
        learning_rate=init_lr,
		train_batch_size=8,
		save_eval_checkpoints=False,
		save_model_every_epoch=False,
    )

    model = ClassificationModel(
        mname, models[mname],
        args=model_args,
        num_labels=2,
        weight=class_weight,
        use_cuda=True,
    )

    model.train_model(train_df)
    result, model_outputs, wrong_predictions = model.eval_model(test_df)

    tp = result['tp']
    tn = result['tn']
    fp = result['fp']
    fn = result['fn']

    precision = 0 if tp==0 and fp==0 else tp / (tp + fp) 
    recall = 0 if tp==0 and fn==0 else tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    if precision == 0 and recall == 0:
        f1 = 0
        f2 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
        f2 = 5*precision*recall/(4*precision+recall)

    tpr = recall
    tnr = 0 if tn==0 and fp==0 else tn/(tn+fp) 
    gmean = sqrt(tpr*tnr)

    out = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'gmean': gmean,
        'accuracy': accuracy
    }



def loadData(i):
    path_head = f'../dataset/{i}'
    class_names = ['fair', 'unfair']

    pos = len(list(glob(f'{path_head}/train/unfair/*')))
    neg = len(list(glob(f'{path_head}/train/fair/*')))

    train_texts, train_labels = read_labeled_data(f'{path_head}/train', class_names, pos='unfair')
    test_texts, test_labels = read_labeled_data(f'{path_head}/test', class_names, pos='unfair')

    train_df = pd.DataFrame(list(zip(train_texts, train_labels)))
    test_df = pd.DataFrame(list(zip(test_texts, test_labels)))

    return train_df, test_df, pos, neg


def read_labeled_data(dpath, class_names, pos='unfair'):
    texts, labels = [], []
    for label_dir in class_names:
        for text_file in glob(f'{dpath}/{label_dir}/*'):
            text_tmp = []
            with open(text_file, 'r') as r:
                for line in r:
                    text_tmp.append(line.strip())
            texts.append(' '.join(text_tmp))
            labels.append(1 if label_dir == pos else 0)
    return texts, np.array(labels, dtype='float32')





if __name__ == "__main__":
    main()
