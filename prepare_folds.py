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
parser.add_argument('model')
parser.add_argument('k', help='k for k-fold cross-validation',
                    default='10', type=int)
args = parser.parse_args()

if not args.model:
    print("You need to provide a path to the model to save or load it.")
    sys.exit()

if not args.k:
    print("You need to provide the number of folds.")
    sys.exit()


instances = []
with open(args.model + '/features.tsv', 'r', encoding='utf8') as f:
    next(f)  # Skip header
    for line in f:
        instances.append(line)


kfold = model_selection.KFold(args.k, shuffle=True)

for fold_nr, (train_idx, test_idx) in enumerate(kfold.split(instances)):
    print('Fold #' + str(fold_nr))
    folder = '{}/fold-{}/'.format(args.model, fold_nr)
    Path(folder).mkdir(parents=True, exist_ok=True)
    train_idx.tofile(folder + 'folds-train.npy')
    test_idx.tofile(folder + 'folds-test.npy')
    with open(folder + 'train_data.txt', 'w+', encoding='utf8') as f:
        for idx in train_idx:
            f.write(instances[idx])
    with open(folder + 'test_data.txt', 'w+', encoding='utf8') as f:
        for idx in test_idx:
            f.write(instances[idx])
