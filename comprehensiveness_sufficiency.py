# coding: utf-8

import argparse
import datetime
import pickle
from predict_fold import get_features
from predict import encode, predict_proba


parser = argparse.ArgumentParser()
parser.add_argument('dir', help='path to the model + fold')
parser.add_argument('type', help='type of the data (dialects/tweets)')
parser.add_argument('--mode', dest='mode', help='all/pos', default='all')
parser.add_argument('--top', dest='n_top_features', default='[5,10,20]')
args = parser.parse_args()

labels = ['0', '1'] if args.type == 'tweets' else ['oestnorsk', 'vestnorsk',
                                                   'nordnorsk', 'troendersk']
n_labels = len(labels)
linear_svc = args.type == 'dialects'

n_top_features = [int(n.strip()) for n in args.n_top_features[1:-1].split(',')]

print("Loading the model")
with open('{}/model-files/classifier.pickle'.format(args.dir), 'rb') as file:
    classifier = pickle.load(file)

raw_train, ngrams_train, labels_train = get_features(
    '{}/train_data.txt'.format(args.dir))
raw_test, ngrams_test, labels_test = get_features(
    '{}/test_data.txt'.format(args.dir))
train_x, test_x, train_y, test_y, label_encoder, vectorizer = encode(
        ngrams_train, ngrams_test, labels_train, labels_test)

scores = {}
for label in labels:
    label_scores = []
    with open('{}/importance_values_{}.txt'.format(args.dir, label),
              encoding='utf8') as f:
        prev_idx = None
        values = []
        for line in f:  # ends with an empty line!
            sample_idx, feature, score = line.strip().split('\t')
            if prev_idx and prev_idx != sample_idx:
                label_scores.append(values)
                values = []
            prev_idx = sample_idx
            values.append((feature, float(score)))
        if len(label_scores) == int(prev_idx):
            # 0-indexed entries
            # add last line if necessary
            label_scores.append(values)
    scores[label] = label_scores


def get_scores(idx, label, y_pred_enc, y_pred, pred_proba_all, investigating):
    pred_proba_all = predict_proba(classifier, [ngrams], vectorizer,
                                   linear_svc, n_labels)[0]
    investigating_enc = label_encoder.transform([investigating])[0]
    pred_proba = pred_proba_all[investigating_enc]

    if idx % 50 == 0:
        print("- Investigating: {} ({:.2f})".format(investigating, pred_proba))

    importance_scores = sorted(scores[investigating][idx], key=lambda x: x[1],
                               reverse=True)
    for n in n_top_features:
        most_important = []
        for (feature, _) in importance_scores[:n]:
            most_important.append(feature)
        most_important = ' '.join(most_important)

        only_less_important = ' '.join([ngram for ngram in ngrams.split()
                                        if ngram not in most_important])

        pred_proba_less_imp = predict_proba(classifier, [only_less_important],
                                            vectorizer, linear_svc,
                                            n_labels)[0, investigating_enc]
        pred_proba_only_top = predict_proba(classifier, [most_important],
                                            vectorizer, linear_svc,
                                            n_labels)[0, investigating_enc]
        # Comprehensiveness: (Mathew et al., 2020)
        # Removing the features from the actual input should affect the output
        # probability
        comprehensiveness = pred_proba - pred_proba_less_imp
        comp_scores[n].append(comprehensiveness)
        # Sufficiency: (Mathew et al., 2020)
        # Does only using the top _ features get you close to the same output
        # probability?
        sufficiency = pred_proba - pred_proba_only_top
        suff_scores[n].append(sufficiency)
        if idx % 50 == 0:
            print("--- Top {} features".format(n))
            print("----- Most important features:", most_important)
            print("----- Comprehensiveness: {:.2f} ({:.2f} - {:.2f})"
                  .format(comprehensiveness, pred_proba, pred_proba_less_imp))
            print("----- Sufficiency: {:.2f} ({:.2f} - {:.2f})\n"
                  .format(sufficiency, pred_proba, pred_proba_only_top))


comp_scores = {n: [] for n in n_top_features}
suff_scores = {n: [] for n in n_top_features}
for idx, (raw, ngrams, x, label) in enumerate(zip(raw_test, ngrams_test,
                                                  test_x, labels_test)):
    y_pred_enc = classifier.predict(x)[0]
    y_pred = label_encoder.inverse_transform([y_pred_enc])[0]
    pred_proba_all = predict_proba(classifier, [ngrams], vectorizer,
                                   linear_svc, n_labels)[0]
    if idx % 50 == 0:
        print(idx)
        print(datetime.datetime.now())
        print('"' + raw + '""')
        print("Actual: {} ({:.2f}) | Expected: {}"
              .format(y_pred, pred_proba_all[y_pred_enc], label))

    if args.mode == 'all':
        for label in labels:
            get_scores(idx, label, y_pred_enc, y_pred, pred_proba_all, label)
    else:
        get_scores(idx, label, y_pred_enc, y_pred, pred_proba_all, y_pred)


with open('{}/comprehensiveness-sufficiency-{}.tsv'
          .format(args.dir, args.mode), 'w+', encoding='utf8') as file:
    for n in n_top_features:
        mean_comp = sum(comp_scores[n]) / len(comp_scores[n])
        file.write("Comprehensiveness\t{}\t{:.4f}\n".format(n, mean_comp))
        mean_suff = sum(suff_scores[n]) / len(suff_scores[n])
        file.write("Sufficiency\t{}\t{:.4f}\n".format(n, mean_suff))
        print("Top {} features".format(n))
        print("-- Mean comprehensiveness: {:.2f}".format(mean_comp))
        print("-- Mean sufficiency: {:.2f}".format(mean_suff))
