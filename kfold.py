# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ngram_lime.lime.lime_text import LimeTextExplainer
import re
import sys
import argparse
import pickle
import datetime
from predict import *
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, help="'dialects' or 'tweets'")
parser.add_argument('model')
parser.add_argument('--k', dest='k', help='k for k-fold cross-validation',
                    default='10', type=int)
args = parser.parse_args()

if args.type == 'dialects':
    DIALECTS = True
elif args.type == 'tweets':
    DIALECTS = False
else:
    print("Expected 'dialects' or 'tweets'")
    sys.exit()

if not args.model:
    print("You need to provide a path to the model to save or load it")
    sys.exit()

def get_features(filename):
    raw, ngrams, labels = [], [], []
    with open(filename, 'r', encoding='utf8') as f:
        next(f)  # Skip header
        for line in f:
            cells = line.strip().split('\t')
            # Utterance, label, various encoding levels
            raw.append(cells[0])
            labels.append(cells[1])
            ngrams.append(' '.join(cells[2:]))
    return np.array(raw), np.array(ngrams), np.array(labels)


raw, ngrams, labels = get_features(args.model + '/features.tsv')
n_labels = len(set(labels))
class_weight = None if DIALECTS else {0: 1, 1: 2}

with open(args.model + '/folds.txt', 'w+', encoding='utf8') as f:
    f.write(str(args.k) + ' folds.\n')
kfold = model_selection.KFold(args.k, shuffle=True)

for fold_nr, (train_idx, test_idx) in enumerate(kfold.split(labels)):
    folder = args.model + '/fold-' + str(fold_nr) + '/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    print("\n\nFOLD NUMBER " + str(fold_nr) + "\n")
    print("Preparing the data")
    with open(args.model + '/folds.txt', 'a', encoding='utf8') as f:
        f.write('\nfold ' + str(fold_nr) + '\n')
        f.write('train\n' + str(train_idx) + '\n')
        f.write('test\n' + str(test_idx) + '\n')
    raw_train, raw_test = raw[train_idx], raw[test_idx]
    ngrams_train, ngrams_test = ngrams[train_idx], ngrams[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]
    train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
        ngrams_train, ngrams_test, labels_train, labels_test)

    print("Training the model")
    classifier = train(train_x, train_y, linear_svc=DIALECTS,
                       class_weight=class_weight)
    print("Scoring the model")
    pred = predict(classifier, test_x)
    (acc, f1, conf) = score(pred, test_y)
    print('Accuracy', acc)
    print('F1 macro', f1)
    print('Confusion matrix')
    print(conf)
    with open(folder + 'log.txt'.format(fold_nr), 'w+', encoding='utf8') as f:
        f.write('Train {} ({}, {}) / test {} ({}, {})\n'.format(
            train_x.shape, len(raw_train), len(train_y),
            test_x.shape, len(raw_test), len(test_y)))
        f.write('Accuracy\t{:.4f}\n'.format(acc))
        f.write('F1 macro\t{:.4f}\n'.format(f1))
        f.write('Confusion matrix\n')
        f.write(str(conf) + '\n')
    print('Generating explanations')
    explain_lime(classifier, vectorizer, label_encoder, n_labels, raw_test,
                 ngrams_test, test_x, test_y, folder, linear_svc=DIALECTS)            
