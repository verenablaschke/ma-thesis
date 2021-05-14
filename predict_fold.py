# coding: utf-8

import numpy as np
import argparse
import datetime
import pickle
from predict import encode, encode_embeddings, score, train, \
                    explain_lime, get_features
from pathlib import Path
from transformers import FlaubertModel, FlaubertTokenizer, \
                         BertTokenizer, BertModel


def get_data_for_fold(args, folder, n_bpe_toks):
    print("Encoding the data.")
    raw_train, ngrams_train, labels_train = get_features(
        '{}/train_data.txt'.format(folder))
    raw_test, ngrams_test, labels_test = get_features(
        '{}/test_data.txt'.format(folder))
    n_labels = len(set(labels_train))
    class_weight = {0: 1, 1: 2} if args.type == 'tweets' else None
    flaubert, flaubert_tokenizer = None, None

    if args.use_embeddings:
        if not args.load_embeddings:
            if 'flaubert' in args.embedding_model:
                bert_model, _ = FlaubertModel.from_pretrained(
                    args.embedding_model, output_loading_info=True)
                tokenizer = FlaubertTokenizer.from_pretrained(
                    args.embedding_model,
                    do_lowercase='uncased' in args.embedding_model)
            else:
                bert_model = BertModel.from_pretrained(
                    args.embedding_model, output_loading_info=True)
                tokenizer = BertTokenizer.from_pretrained(
                    args.embedding_model,
                    do_lowercase='uncased' in args.embedding_model)
        embedding_size = 768
        if '_small_' in args.embedding_model:
            embedding_size = 512
        elif '_large_' in args.embedding_model:
            embedding_size = 1024
        train_x, test_x, train_y, test_y, label_encoder = encode_embeddings(
            ngrams_train, ngrams_test, labels_train, labels_test,
            tokenizer, bert_model, n_bpe_toks,
            args.load_embeddings,
            args.flaubert_micro_batch_size, args.flaubert_macro_batch_size,
            args.flaubert_macro_batch_start,
            folder, embedding_size, args.embedding_model,
            flatten=args.model_type == 'svm')
        vectorizer = None
    else:
        train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
            ngrams_train, ngrams_test, labels_train, labels_test)

    return raw_train, ngrams_train, labels_train, \
        raw_test, ngrams_test, labels_test, \
        n_labels, class_weight, flaubert, flaubert_tokenizer, \
        train_x, test_x, train_y, test_y, label_encoder, vectorizer


def predict_fold(
                 # always the same:
                 args,
                 # the same for one fold and data encoding setup:
                 fold, folder,
                 raw_train, ngrams_train, labels_train,
                 raw_test, ngrams_test, labels_test,
                 n_labels, class_weight, flaubert,
                 flaubert_tokenizer, train_x, test_x, train_y, test_y,
                 label_encoder, vectorizer,
                 # specific initialization:
                 model_type, hidden, epochs, batch_size,
                 learning_rate, dropout_rate
                 ):
    MODEL_FILES_FOLDER = '{}/model-files/'.format(folder)
    LIME_FOLDER = folder + args.output_subfolder + '/'
    Path(LIME_FOLDER).mkdir(parents=True, exist_ok=True)
    LOG_FILE = LIME_FOLDER + 'log.txt'
    dropout_percentage = int(100 * dropout_rate)
    lr_int = int(1000 * learning_rate)
    n_bpe_toks = train_x.shape[-2]
    embedding_model_name = args.embedding_model.split('/')[-1]
    FILE_SFX = '-{}-h{}-b{}-d{}-ep{}-T{}-em{}-lr{}'.format(
        model_type, hidden, batch_size, dropout_percentage, epochs,
        n_bpe_toks, embedding_model_name, lr_int)
    if args.explicit_log:
        LOG_FILE = '{}log{}.txt'.format(LIME_FOLDER, FILE_SFX)

    print('Current fold: {}'.format(fold))
    print('Current model_type: {}'.format(model_type))
    print('Current hidden: {}'.format(hidden))
    print('Current epochs: {}'.format(epochs))
    print('Current batch_size: {}'.format(batch_size))
    print('Current learning_rate: {}'.format(learning_rate))
    print('Current dropout_rate: {}'.format(dropout_rate))
    print('Current input dimension: {}'.format(train_x.shape))
    for arg, val in vars(args).items():
        print('{}: {}'.format(arg, val))
    with open(LOG_FILE, 'w+', encoding='utf8') as f:
        f.write('{}\nArguments:\n'.format(datetime.datetime.now()))
        f.write('Current fold: {}\n'.format(fold))
        f.write('Current model_type: {}\n'.format(model_type))
        f.write('Current hidden: {}\n'.format(hidden))
        f.write('Current epochs: {}\n'.format(epochs))
        f.write('Current batch_size: {}\n'.format(batch_size))
        f.write('Current learning_rate: {}\n'.format(learning_rate))
        f.write('Current dropout_rate: {}\n'.format(dropout_rate))
        f.write('Current input dimension: {}\n'.format(train_x.shape))
        for arg, val in vars(args).items():
            f.write('{}: {}\n'.format(arg, val))

    print(datetime.datetime.now())
    if args.load_model:
        print("Loading the model")
        with open(MODEL_FILES_FOLDER + 'classifier.pickle', 'rb') as file:
            classifier = pickle.load(file)
    else:
        print("Training the model")
        start_time = datetime.datetime.now()
        classifier = train(train_x, train_y, model_type=model_type,
                           n_classes=4 if args.type == 'dialects' else 2,
                           linear_svc=args.type == 'dialects',
                           class_weight=class_weight, verbose=args.verbose,
                           log_file=LOG_FILE, hidden_size=hidden,
                           epochs=epochs, batch_size=batch_size,
                           learning_rate=learning_rate,
                           dropout_rate=dropout_rate)
        done_time = datetime.datetime.now()

        print("Scoring the model")
        if 'attn' in model_type:
            pred, attn_scores = classifier.predict(test_x)
        else:
            pred = classifier.predict(test_x)
        if 'nn' in model_type:
            pred = np.argmax(pred, axis=1)
        (acc, f1, conf) = score(pred, test_y)
        print('Accuracy', acc)
        print('F1 macro', f1)
        print('Confusion matrix')
        print(conf)
        with open(LOG_FILE, 'a', encoding='utf8') as f:
            f.write('Train {} ({}, {}) / test {} ({}, {})\n'.format(
                train_x.shape, len(raw_train), len(train_y),
                test_x.shape, len(raw_test), len(test_y)))
            f.write('Trained the model from {} to {}\n'.format(start_time,
                                                               done_time))
            f.write('Accuracy\t{:.4f}\n'.format(acc))
            f.write('F1 macro\t{:.4f}\n'.format(f1))
            f.write('Confusion matrix\n' + str(conf) + '\n')

    if args.save_model:
        print("Saving model")
        Path(MODEL_FILES_FOLDER).mkdir(parents=True, exist_ok=True)
        with open(MODEL_FILES_FOLDER + 'classifier.pickle', 'wb') as file:
            pickle.dump(classifier, file)

    if not args.fit_model_only:
        print('Generating explanations')
        if 'attn' in model_type:
            with open('{}attention_scores{}.txt'.format(LIME_FOLDER, FILE_SFX),
                      'w+', encoding='utf8') as f:
                f.write('LABEL\tPRED\tATTN{}\tTOKENS{}\n'.format(
                    '\t' * (n_bpe_toks - 1),
                    '\t' * (n_bpe_toks - 1)))
                for i, (x, y_true, y_pred, attn) in enumerate(
                        zip(ngrams_test, test_y, pred, attn_scores)):
                    x = x.split(' ')
                    filler = (n_bpe_toks - len(x)) * '\t<FILLER>'
                    tokens = '\t'.join(x[:n_bpe_toks - 1] + [x[-1]]) + filler
                    attention_score = '\t'.join(str(a[0]) for a in attn)
                    # if i % 100 == 0:
                    #     print('{}\t{}\t{}\t{}\n'.format(
                    #         y_true, y_pred, attention_score, tokens))

                    f.write('{}\t{}\t{}\t{}\n'.format(
                        y_true, y_pred, attention_score, tokens))
        else:
            explain_lime(classifier, vectorizer, label_encoder, n_labels,
                         raw_test,
                         ngrams_test, test_x, test_y, LIME_FOLDER,
                         args.n_lime_features,
                         args.n_lime_samples, args.type == 'dialects',
                         model_type == 'nn',
                         args.recalculate_ngrams,
                         flaubert, flaubert_tokenizer, n_bpe_toks)
    return acc, f1, FILE_SFX


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the model')
    parser.add_argument('type', choices=['dialects', 'tweets'])
    parser.add_argument('folds', help='fold numbers', nargs='+')
    parser.add_argument('--mlm', dest='model_type',
                        help='type of the ML model',
                        choices=['svm',
                                 'rnn', 'rnn-attn', 'rnn-uniform-attn',
                                 'ffnn', 'ffnn-attn', 'ffnn-uniform-attn'],
                        default=['svm'], type=str, nargs='+')
    parser.add_argument('--embed', dest='use_embeddings', default=False,
                        action='store_true')
    parser.add_argument('--embmod', dest='embedding_model',
                        choices=['flaubert/flaubert_base_cased',
                                 'flaubert/flaubert_small_cased',
                                 'flaubert/flaubert_large_cased',
                                 'bert-base-multilingual-cased'],
                        default='flaubert/flaubert_base_cased', type=str)
    parser.add_argument('--emblen', dest='n_bpe_toks', default=[20], type=int,
                        nargs='+')
    parser.add_argument('--embbatch', dest='flaubert_micro_batch_size',
                        default=50, type=int)
    parser.add_argument('--embbmacro', dest='flaubert_macro_batch_size',
                        default=2000, type=int)
    parser.add_argument('--embbmacrostart', dest='flaubert_macro_batch_start',
                        default=0, type=int)
    parser.add_argument('--limefeat', dest='n_lime_features', default=100,
                        type=int)
    parser.add_argument('--z', dest='n_lime_samples', default=1000, type=int)
    parser.add_argument('--save', dest='save_model', default=False,
                        action='store_true')
    parser.add_argument('--load', dest='load_model', default=False,
                        action='store_true')
    parser.add_argument('--load_emb', dest='load_embeddings',
                        default=False, action='store_true')
    parser.add_argument('--recalc', dest='recalculate_ngrams', default=False,
                        help='no overlapping LIME features',
                        action='store_true')
    parser.add_argument('--v', dest='verbose', default=False,
                        action='store_true')
    parser.add_argument('--out', dest='output_subfolder', default='', type=str)
    parser.add_argument('--h', dest='hidden', default=[512], type=int,
                        nargs='+')
    parser.add_argument('--ep', dest='epochs', default=[10], type=int,
                        nargs='+')
    parser.add_argument('--b', dest='batch_size', default=[128], type=int,
                        nargs='+')
    parser.add_argument('--lr', dest='learning_rate', default=[0.001],
                        type=float, nargs='+')
    parser.add_argument('--drop', dest='dropout_rate', default=[0.2], type=float,
                        nargs='+')
    parser.add_argument('--nolime', dest='fit_model_only', default=False,
                        action='store_true')
    parser.add_argument('--log', dest='explicit_log',
                        help='add parameters to log file name', default=False,
                        action='store_true')
    args = parser.parse_args()

    for fold in args.folds:
        for emblen in args.n_bpe_toks:
            folder = '{}/fold-{}/'.format(args.model, fold)
            (raw_train, ngrams_train, labels_train,
                raw_test, ngrams_test, labels_test,
                n_labels, class_weight,
                flaubert, flaubert_tokenizer,
                train_x, test_x,
                train_y, test_y,
                label_encoder, vectorizer) = get_data_for_fold(args, folder,
                                                               emblen)
            accuracies = []
            f1_scores = []
            for model_type in args.model_type:
                for hidden in args.hidden:
                    for epochs in args.epochs:
                        for batch_size in args.batch_size:
                            for learning_rate in args.learning_rate:
                                for dropout_rate in args.dropout_rate:
                                    acc, f1, filename = predict_fold(
                                         # always the same:
                                         args,
                                         # the same for one fold:
                                         fold, folder,
                                         raw_train, ngrams_train, labels_train,
                                         raw_test, ngrams_test, labels_test,
                                         n_labels, class_weight,
                                         flaubert, flaubert_tokenizer,
                                         train_x, test_x,
                                         train_y, test_y,
                                         label_encoder, vectorizer,
                                         # specific initialization:
                                         model_type, hidden, epochs,
                                         batch_size, learning_rate,
                                         dropout_rate)
                                    accuracies.append((acc, filename))
                                    f1_scores.append((f1, filename))

        print('Accuracy')
        with open(folder + 'log_acc.tsv', 'w+', encoding='utf8') as f:
            for i, (acc, run) in enumerate(sorted(accuracies, reverse=True)):
                if i < 5:
                    print(acc, run)
            f.write('{}\t{}\n'.format(acc, run))

        print('\nF1')
        with open(folder + 'log_f1.tsv', 'w+', encoding='utf8') as f:
            for i, (f1, run) in enumerate(sorted(f1_scores, reverse=True)):
                if i < 5:
                    print(f1, run)
            f.write('{}\t{}\n'.format(f1, run))
