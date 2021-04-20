# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

from glob import glob
import pickle



def main():
    for n in [1, 2]:
        gen(n, anonym=True, anonym_plus=True)


def gen(n=1, anonym=True, anonym_plus=True):
    dataset = loadData('./data/Sentences', anonym=anonym)
    labels = loadData('./data/Labels', label=True)

    corpus = []
    for d in dataset:
        corpus.extend(d)

    vectorizer = TFIDF(ngram_range=(1, n), min_df=3)
    vectorizer.fit(corpus)

    out = []
    for d, y in zip(dataset, labels):
        X = vectorizer.transform(d)

        out.append((X, y))

    out_path = './vectors/tfidf_min2'
    if n == 1:
        out_path = f'{out_path}_uni'
    else:
        out_path = f'{out_path}_unibi'

    if anonym:
        out_path = f'{out_path}_anonym'

    if anonym_plus:
        out_path = f'{out_path}_plus'

    with open(f'{out_path}.dump', 'wb') as w:
        pickle.dump(out, w)



def loadData(base_path, anonym=True, label=False):
    out = []
    for path in glob(f'{base_path}/*'):
        print(path)

        out_tmp = []
        with open(path, 'r') as r:
            if label:
                out_tmp = [line.strip() for line in r]

            elif anonym:
                service_name = path[path.rfind('/')+1: path.find('.txt')].lower() 

                for line in r:
                    tmp = line.strip()
                    tmp = tmp.replace(service_name, 'SERVICE_NAME')
                    tmp = tmp.replace('-lrb-', '')
                    tmp = tmp.replace('-llb-', '')
                    tmp = tmp.replace('-rrb-', '')
                    tmp = tmp.replace('-lsb-', '')
                    tmp = tmp.replace('-rsb-', '')
                    out_tmp.append(tmp)

            else:
                out_tmp = [line.strip() for line in r]

        out.append(out_tmp)
    return out 


if __name__ == "__main__":
    main()
