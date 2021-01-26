# coding: utf-8

# called by kfold.py

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


def split_ngrams(joined_ngrams):
    # Used by the TfidfVectorizer, important for the modified LIME
    return joined_ngrams.split(' ')


def encode(ngrams_train, ngrams_test, labels_train, labels_test,
           max_features=5000):

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(labels_train)
    test_y = label_encoder.transform(labels_test)

    vectorizer = TfidfVectorizer(max_features=max_features,
                                 analyzer=split_ngrams)
    train_x = vectorizer.fit_transform(ngrams_train)
    test_x = vectorizer.transform(ngrams_test)

    return train_x, test_x, train_y, test_y, label_encoder, vectorizer


def preprocess_and_vectorize(utterance, vectorizer):
    return vectorizer.transform([utterance])
    # return vectorizer.transform([preprocess(utterance)])


def train(train_x, train_y, linear_svc, class_weight=None):
    if linear_svc:
        model = svm.LinearSVC(C=1.0, class_weight=class_weight)
    else:
        # Binary cases
        model = svm.SVC(C=1.0, probability=True, class_weight=class_weight)
    model.fit(train_x, train_y)
    return model


def predict(model, test_x):
    return model.predict(test_x)


def predict_instance(model, utterance, label_encoder, vectorizer, linear_svc):
    # x = vectorizer.transform([preprocess(utterance)])
    x = vectorizer.transform([utterance])
    pred = model.predict(x)
    margins = model.decision_function(x)
    if linear_svc:
        exp = np.exp(margins)
        softmax = exp / np.sum(exp)
    #     print("{}: {}".format(utterance, softmax.round(3)))
        return pred[0], label_encoder.inverse_transform(pred)[0], margins, softmax
    return pred[0], label_encoder.inverse_transform(pred)[0], margins, model.predict_proba(x)

def predict_proba2(model, data, vectorizer, linear_svc, n_labels=4):
    probs = np.zeros((len(data), n_labels))
    for i, utterance in enumerate(data):
        x = vectorizer.transform([utterance])
        if linear_svc:
            pred = model.predict(x)
            margins = model.decision_function(x)
            exp = np.exp(margins)
            probs[i] = exp / np.sum(exp)  # softmax
        else:
            probs[i] = model.predict_proba(x)
    GET_NGRAMS = True
    return probs


def score(pred, test_y):
    return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro'), confusion_matrix(test_y, pred)


def support_vectors(model, label_encoder, train_x, train_x_raw,
                    labels=[0, 1, 2, 3]):
    for label in labels:
        print(label_encoder.inverse_transform([label])[0])
        print('==========================\n')
        dec_fn = model.decision_function(train_x)[:, label]
        # support vectors and vectors 'in the middle of the street'
        support_vector_indices_pos = np.where(np.logical_and(dec_fn > 0, dec_fn <= 1))[0]
        print('positive side')
        for idx in support_vector_indices_pos:
            print('-', train_x_raw[idx])
        support_vector_indices_neg = np.where(np.logical_and(dec_fn <= 0, dec_fn >= -1))[0]
        print()
        print('negative side')
        for idx in support_vector_indices_pos:
            print('-', train_x_raw[idx])
        print('\n\n')


def instances_far_from_decision_boundary(model, label_encoder, train_x,
                                         train_x_raw, labels=[0, 1, 2, 3]):
    for label in labels:
        print(label_encoder.inverse_transform([label])[0])
        print('==========================\n')
        dec_fn = model.decision_function(train_x)[:, label]
        # vectors far away from the decision boundary
        support_vector_indices_pos = np.where(dec_fn > 3)[0]
        print('positive side')
        for idx in support_vector_indices_pos:
            print(dec_fn[idx].round(3), train_x_raw[idx])
    #     support_vector_indices_neg = np.where(dec_fn < -3)[0]
    #     print()
    #     print('negative side')
    #     for idx in support_vector_indices_pos:
    #         print('-', train_x_raw[idx])
        print('\n\n')


def save_to_file(filename, nr, label_encoder, lime_results):
    for label, results in lime_results.items():
        with open(filename.format(label_encoder.inverse_transform([label])[0]),
                  'a+', encoding='utf8') as f:
            for feature, score in results:
                f.write('{}\t{}\t{:.10f}\n'.format(nr, feature, score))


def explain_lime(classifier, vectorizer, label_encoder, n_labels, test_x_raw,
                 test_x_ngrams, test_x, test_y, out_folder, linear_svc):
    labels = list(range(n_labels))
    explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform(labels),
                                  split_expression=split_ngrams,
                                  random_state=42,
                                  bow=True, ngram_lvl=True,
                                  utterance2ngrams=split_ngrams,
                                  recalculate_ngrams=False)
    lime_results = {i: [] for i in range(n_labels)}
    with open(out_folder + 'predictions.tsv', 'w+', encoding='utf8') as f_out:
        for i, (utterance, ngrams, encoded) in enumerate(zip(test_x_raw,
                                                             test_x_ngrams,
                                                             test_x)):
            label = test_y[i]
            exp = explainer.explain_instance(ngrams,
                                             lambda z: predict_proba2(classifier,
                                                                      z,
                                                                      vectorizer,
                                                                      linear_svc,
                                                                      n_labels),
                                             num_features=20,
                                             labels=labels,
                                             # labels=interesting_labels
                                             num_samples=1000
                                             )
            for lab in labels:
                lime_results[lab] = lime_results[lab] + exp.as_list(label=lab)

            pred = classifier.predict(encoded)
            f_out.write(utterance + '\t')
            f_out.write(label_encoder.inverse_transform([label])[0] + '\t')
            f_out.write(label_encoder.inverse_transform(pred)[0] + '\n')

            if i % 50 == 0:
                interesting_labels = [label]
                pred = predict_instance(classifier, ngrams, label_encoder,
                                        vectorizer, linear_svc)
                now = datetime.datetime.now()
                print(i)
                print(now)
                print('"' + utterance + '""')
                print('ACTUAL', label_encoder.inverse_transform([label])[0])
                print('PREDICTED', pred[1])
                if pred[0] not in interesting_labels:
                    interesting_labels.append(pred[0])
                # for l in interesting_labels:
                #     exp.show_in_notebook(text=utterance, labels=(l,))
                print('\n')
                save_to_file(out_folder + '/importance_values_{}.txt', i,
                             label_encoder, lime_results)
                with open(out_folder + '/log.txt', 'a', encoding='utf8') as f:
                    f.write(str(i) + '  --  ' + str(now) + '  --  "' + utterance + '""\n')
                    f.write('ACTUAL: ' + str(label_encoder.inverse_transform([label])[0]) + '\n')
                    f.write('PREDICTED: ' + str(pred[1]) + '\n')
                    for label_nr in range(n_labels):
                        lab = label_encoder.inverse_transform([label_nr])[0]
                        f.write('Class ' + lab + ': ' + ', '.join(
                            ['{}\t{:.5f}\n'.format(x[0], x[1]) for x in exp.as_list(label=0)[:5]]) + '\n')
                    f.write('\n')
                lime_results = {i: [] for i in range(n_labels)}
            i += 1

    i -= 1
    if i % 50 != 0:
        save_to_file(out_folder + '/importance_values_{}.txt', i, label_encoder,
                     lime_results)
