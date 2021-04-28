# coding: utf-8

import numpy as np
import argparse
import datetime
import pickle
from predict import encode, encode_embeddings, score, train, \
                    explain_lime
from pathlib import Path
from transformers import FlaubertModel, FlaubertTokenizer


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the model')
    parser.add_argument('type', choices=['dialects', 'tweets'])
    parser.add_argument('fold', help='fold number')
    parser.add_argument('--mlm', dest='model_type',
                        help='type of the ML model',
                        choices=['svm', 'nn', 'nn-attn'],
                        default='svm', type=str)
    parser.add_argument('--embed', dest='use_embeddings', default=False,
                        action='store_true')
    parser.add_argument('--embmod', dest='embedding_model',
                        default='flaubert/flaubert_base_cased', type=str)
    parser.add_argument('--emblen', dest='n_bpe_toks', default=20, type=int)
    parser.add_argument('--embbatch', dest='flaubert_batch_size', default=50,
                        type=int)
    parser.add_argument('--limefeat', dest='n_lime_features', default=100,
                        type=int)
    parser.add_argument('--z', dest='n_lime_samples', default=1000, type=int)
    parser.add_argument('--save', dest='save_model', default=False,
                        action='store_true')
    parser.add_argument('--load', dest='load_model', default=False,
                        action='store_true')
    parser.add_argument('--recalc', dest='recalculate_ngrams', default=False,
                        help='no overlapping LIME features',
                        action='store_true')
    parser.add_argument('--v', dest='verbose', default=False,
                        action='store_true')
    parser.add_argument('--out', dest='output_subfolder', default='', type=str)
    parser.add_argument('--h', dest='hidden', default=512, type=int)
    parser.add_argument('--ep', dest='epochs', default=10, type=int)
    parser.add_argument('--b', dest='batch_size', default=128, type=int)
    parser.add_argument('--lr', dest='learning_rate', default=0.001,
                        type=float)
    parser.add_argument('--drop', dest='dropout_rate', default=0.2, type=float)
    parser.add_argument('--nolime', dest='fit_model_only', default=False,
                        action='store_true')
    parser.add_argument('--log', dest='explicit_log',
                        help='add parameters to log file name', default=False,
                        action='store_true')
    args = parser.parse_args()

    folder = '{}/fold-{}/'.format(args.model, args.fold)
    MODEL_FILES_FOLDER = '{}/model-files/'.format(folder)
    LIME_FOLDER = folder + args.output_subfolder + '/'
    Path(LIME_FOLDER).mkdir(parents=True, exist_ok=True)
    LOG_FILE = LIME_FOLDER + 'log.txt'
    dropout_percentage = int(100 * args.dropout_rate)
    if args.explicit_log:
        LOG_FILE = '{}log-{}-h{}-b{}-d{}-ep{}-em{}.txt'.format(
            LIME_FOLDER, args.model_type, args.hidden, args.batch_size,
            dropout_percentage, args.epochs, args.n_bpe_toks)

    for arg, val in vars(args).items():
        print('{}: {}'.format(arg, val))
    with open(LOG_FILE, 'w+', encoding='utf8') as f:
        f.write('{}\nArguments:\n'.format(datetime.datetime.now()))
        for arg, val in vars(args).items():
            f.write('{}: {}\n'.format(arg, val))

    print("Encoding the data.")
    raw_train, ngrams_train, labels_train = get_features(
        '{}/train_data.txt'.format(folder))
    raw_test, ngrams_test, labels_test = get_features(
        '{}/test_data.txt'.format(folder))
    n_labels = len(set(labels_train))
    class_weight = {0: 1, 1: 2} if args.type == 'tweets' else None

    if args.use_embeddings:
        flaubert, _ = FlaubertModel.from_pretrained(args.embedding_model,
                                                    output_loading_info=True)
        flaubert_tokenizer = FlaubertTokenizer.from_pretrained(
            args.embedding_model,
            do_lowercase='uncased' in args.embedding_model)
        embedding_size = 768
        if '-small-' in args.embedding_model:
            embedding_size = 512
        elif '-large-' in args.embedding_model:
            embedding_size = 1024
        train_x, test_x, train_y, test_y, label_encoder = encode_embeddings(
            ngrams_train, ngrams_test, labels_train, labels_test,
            flaubert_tokenizer, flaubert, args.n_bpe_toks,
            args.flaubert_batch_size,
            embedding_size, flatten=args.model_type == 'svm')
        vectorizer = None
    else:
        flaubert, flaubert_tokenizer = None, None
        train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
            ngrams_train, ngrams_test, labels_train, labels_test)

    print(datetime.datetime.now())
    if args.load_model:
        print("Loading the model")
        with open(MODEL_FILES_FOLDER + 'classifier.pickle', 'rb') as file:
            classifier = pickle.load(file)
    else:
        print("Training the model")
        start_time = datetime.datetime.now()
        classifier = train(train_x, train_y, model_type=args.model_type,
                           n_classes=4 if args.type == 'dialects' else 2,
                           linear_svc=args.type == 'dialects',
                           class_weight=class_weight, verbose=args.verbose,
                           log_file=LOG_FILE, hidden_size=args.hidden,
                           epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate,
                           dropout_rate=args.dropout_rate)
        done_time = datetime.datetime.now()

        print("Scoring the model")
        if args.model_type == 'nn-attn':
            pred, attn_scores = classifier.predict(test_x)
        else:
            pred = classifier.predict(test_x)
        if 'nn' in args.model_type:
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
        if args.model_type == 'nn-attn':
            with open('{}attention_scores-{}-h{}-b{}-d{}-ep{}-em{}.txt'.format(
                    LIME_FOLDER, args.model_type, args.hidden, args.batch_size,
                    dropout_percentage, args.epochs, args.n_bpe_toks),
                      'w+', encoding='utf8') as f:
                f.write('LABEL\tPRED\tATTN{}\tTOKENS{}\n'.format(
                    '\t' * (args.n_bpe_toks - 1),
                    '\t' * (args.n_bpe_toks - 1)))
                for i, (x, y_true, y_pred, attn) in enumerate(
                        zip(ngrams_test, test_y, pred, attn_scores)):
                    x = x.split(' ')
                    filler = (args.n_bpe_toks - len(x)) * '\t<FILLER>'
                    tokens = '\t'.join(x[:args.n_bpe_toks - 1] + [x[-1]]) + filler
                    attention_score = '\t'.join(str(a[0]) for a in attn)
                    if i % 100 == 0:
                        print('{}\t{}\t{}\t{}\n'.format(
                            y_true, y_pred, attention_score, tokens))

                    f.write('{}\t{}\t{}\t{}\n'.format(
                        y_true, y_pred, attention_score, tokens))
        else:
            explain_lime(classifier, vectorizer, label_encoder, n_labels,
                         raw_test,
                         ngrams_test, test_x, test_y, LIME_FOLDER,
                         args.n_lime_features,
                         args.n_lime_samples, args.type == 'dialects',
                         args.model_type == 'nn',
                         args.recalculate_ngrams,
                         flaubert, flaubert_tokenizer, args.n_bpe_toks)
