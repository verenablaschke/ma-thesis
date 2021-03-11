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
from transformers import FlaubertModel, FlaubertTokenizer
import torch


parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('type', help='type of the data (dialects/tweets)')
parser.add_argument('fold', help='fold number')
parser.add_argument('--embed', dest='use_embeddings', default=False,
                    action='store_true')
parser.add_argument('--lime', dest='n_lime_features', default=100, type=int)
args = parser.parse_args()



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

print("Encoding the data.")
folder = '{}/fold-{}/'.format(args.model, args.fold)
raw_train, ngrams_train, labels_train = get_features(folder + 'train_data.txt')
raw_test, ngrams_test, labels_test = get_features(folder + 'test_data.txt')
n_labels = len(set(labels_train))
class_weight = {0: 1, 1: 2} if args.type == 'tweets' else None

if args.use_embeddings:
    modelname = 'flaubert/flaubert_base_cased' # TODO try out large
    flaubert, _ = FlaubertModel.from_pretrained(modelname,
                                                output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname,
                                                           do_lowercase=False)
    train_x, test_x, train_y, test_y, label_encoder, max_len = encode_embeddings(
        ngrams_train, ngrams_test, labels_train, labels_test,
        flaubert_tokenizer, flaubert, max_len=42)
else:
    flaubert, flaubert_tokenizer, max_len = None, None, None
    train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
        ngrams_train, ngrams_test, labels_train, labels_test)

print("Training the model")
classifier = train(train_x, train_y, linear_svc=args.type == 'dialects',
                   class_weight=class_weight)
print("Scoring the model")
pred = predict(classifier, test_x)
(acc, f1, conf) = score(pred, test_y)
print('Accuracy', acc)
print('F1 macro', f1)
print('Confusion matrix')
print(conf)
with open(folder + 'log.txt', 'w+', encoding='utf8') as f:
    f.write('Train {} ({}, {}) / test {} ({}, {})\n'.format(
        train_x.shape, len(raw_train), len(train_y),
        test_x.shape, len(raw_test), len(test_y)))
    f.write('Accuracy\t{:.4f}\n'.format(acc))
    f.write('F1 macro\t{:.4f}\n'.format(f1))
    f.write('Confusion matrix\n' + str(conf) + '\n')
    f.write('\nLIME features: {}\n'.format(args.n_lime_features))
print('Generating explanations')
explain_lime(classifier, vectorizer, label_encoder, n_labels, raw_test,
             ngrams_test, test_x, test_y, folder, args.n_lime_features,
             args.type == 'dialects',
             flaubert, flaubert_tokenizer, max_len)
