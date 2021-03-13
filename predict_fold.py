# coding: utf-8

import numpy as np
import argparse
import pickle
from predict import encode, encode_embeddings, predict, score, train, \
                    explain_lime
from pathlib import Path
from transformers import FlaubertModel, FlaubertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('type', help='type of the data (dialects/tweets)')
parser.add_argument('fold', help='fold number')
parser.add_argument('--embed', dest='use_embeddings', default=False,
                    action='store_true')
parser.add_argument('--embmod', dest='embedding_model',
                    default='flaubert/flaubert_base_cased' , type=str)
parser.add_argument('--emblen', dest='n_bpe_toks', default=20, type=int)
parser.add_argument('--embbatch', dest='batch_size', default=50, type=int)
parser.add_argument('--limefeat', dest='n_lime_features', default=100,
                    type=int)
parser.add_argument('--z', dest='n_lime_samples', default=1000, type=int)
parser.add_argument('--save', dest='save_model', default=False,
                    action='store_true')
parser.add_argument('--load', dest='load_model', default=False,
                    action='store_true')
parser.add_argument('--out', dest='output_subfolder', default='', type=str)
args = parser.parse_args()


folder = '{}/fold-{}/'.format(args.model, args.fold)
MODEL_FILES_FOLDER = '{}/model-files/'.format(folder)
LIME_FOLDER = folder + args.output_subfolder + '/'


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
raw_train, ngrams_train, labels_train = get_features(folder + 'train_data.txt')
raw_test, ngrams_test, labels_test = get_features(folder + 'test_data.txt')
n_labels = len(set(labels_train))
class_weight = {0: 1, 1: 2} if args.type == 'tweets' else None

if args.use_embeddings:
    flaubert, _ = FlaubertModel.from_pretrained(args.embedding_model,
                                                output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(
        args.embedding_model, do_lowercase='uncased' in args.embedding_model)
    embedding_size = 768
    if '-small-' in args.embedding_model:
        embedding_size = 512
    elif '-large-' in args.embedding_model:
        embedding_size = 1024
    train_x, test_x, train_y, test_y, label_encoder = encode_embeddings(
        ngrams_train, ngrams_test, labels_train, labels_test,
        flaubert_tokenizer, flaubert, args.n_bpe_toks, args.batch_size,
        embedding_size)
    vectorizer = None
else:
    flaubert, flaubert_tokenizer = None, None
    train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
        ngrams_train, ngrams_test, labels_train, labels_test)

if args.load_model:
    print("Loading the model")
    with open(MODEL_FILES_FOLDER + 'classifier.pickle', 'rb') as file:
        classifier = pickle.load(file)
else:
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

if args.save_model:
    print("Saving model")
    Path(MODEL_FILES_FOLDER).mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILES_FOLDER + 'classifier.pickle', 'wb') as file:
        pickle.dump(classifier, file)

print('Generating explanations')
Path(LIME_FOLDER).mkdir(parents=True, exist_ok=True)
explain_lime(classifier, vectorizer, label_encoder, n_labels, raw_test,
             ngrams_test, test_x, test_y, LIME_FOLDER, args.n_lime_features,
             args.n_lime_samples,
             args.type == 'dialects',
             flaubert, flaubert_tokenizer, args.n_bpe_toks)
