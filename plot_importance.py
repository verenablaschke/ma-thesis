import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('type', choices=['dialects', 'tweets'])
parser.add_argument('--m', dest='mode', help='pos / truepos / falsepos / all',
                    default='all')
parser.add_argument('--comb', dest='combination_method',
                    help='options: sqrt (square root of sums), mean',
                    default='sqrt', type=str)
parser.add_argument('--scale', dest='scale_by_model_score',
                    default=False, action='store_true')
parser.add_argument('--label', dest='per_label',
                    default=False, action='store_true')
parser.add_argument('--top', dest='top100',
                    default=False, action='store_true')
args = parser.parse_args()

# Produced by feature_context.py
in_file = '{}/importance-spec-rep-{}-{}-{}scaled.tsv' \
                      .format(args.model, args.mode, args.combination_method,
                              '' if args.scale_by_model_score else 'un')
out_file = '{}/figures/importance-{{}}-{}-{}-{}scaled{}.png' \
                      .format(args.model, args.mode, args.combination_method,
                              '' if args.scale_by_model_score else 'un',
                              '-100' if args.top100 else '')
Path('{}/figures/'.format(args.model)).mkdir(parents=True, exist_ok=True)

labels = ['nordnorsk', 'oestnorsk', 'troendersk', 'vestnorsk'] \
    if args.type == 'dialects' else ['0', '1']

label2features = {}
if args.top100:
    for label in labels:
        label2features[label] = []
        top100_file = '{}/importance_values_{}_{}_{}_{}scaled_sorted_100_context.tsv' \
                      .format(args.model, args.combination_method, label,
                              args.mode,
                              '' if args.scale_by_model_score else 'un')
        with open(top100_file, 'r', encoding='utf8') as f:
            next(f)
            for line in f:
                feature = line.split('\t')[1]
                label2features[label].append(feature)

imp_scores, spec_scores, rep_scores = {}, {}, {}
with open(in_file, 'r', encoding='utf8') as f:
    next(f)  # header
    for line in f:
        cells = line.strip().split('\t')
        feature, label = cells[0], cells[1]
        if args.top100 and feature not in label2features[label]:
            continue
        imp, rep, spec = float(cells[2]), float(cells[3]), float(cells[4])
        try:
            imp_scores[label].append(imp)
            spec_scores[label].append(spec)
            rep_scores[label].append(rep)
        except KeyError:
            imp_scores[label] = [imp]
            spec_scores[label] = [spec]
            rep_scores[label] = [rep]


label2col = {'nordnorsk': 'green', 'troendersk': 'blue',
             'vestnorsk': 'yellow', 'oestnorsk': 'red',
             '0': 'gray', '1': 'red'}
if not args.per_label:
    label2col = {label: 'blue' for label in label2col}

for label in imp_scores.keys():
    imp = np.array(imp_scores[label])
    rep = np.array(rep_scores[label])
    plt.scatter(imp, rep, color=label2col[label], s=3)

importance_label = "Importance ({}, {}, {}scaled by model error)".format(
    args.combination_method, args.mode,
    '' if args.scale_by_model_score else 'un')
plt.xlabel(importance_label)
plt.ylabel("Representativeness")
plt.savefig(out_file.format('rep'))
plt.show()

for label in imp_scores.keys():
    imp = np.array(imp_scores[label])
    spec = np.array(spec_scores[label])
    plt.scatter(imp, spec, color=label2col[label], s=3)

plt.xlabel(importance_label)
plt.ylabel("Specificity")
plt.savefig(out_file.format('spec'))
plt.show()
