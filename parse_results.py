import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('label', help='the class')
parser.add_argument('mode', help='pos / truepos / falsepos / all')
parser.add_argument('k', help='number of folds', type=int)
parser.add_argument('--s', dest='single_fold', help='single fold only',
                    default=False, action='store_true')
parser.add_argument('--min', dest='min_count', help='min. count per fold',
                    default=10, type=int)
parser.add_argument('--comb', dest='combination_method',
                    help='options: sqrt (square root of sums), mean',
                    default='sqrt', type=str)
args = parser.parse_args()

FOLDER = args.model
CLASS = args.label
THRESHOLD = args.min_count
MODE = args.mode


folds = [args.k] if args.single_fold else range(args.k)


scores = {}
for fold in folds:
    if MODE != 'all':
        indices = []
        with open('{}/fold-{}/predictions.tsv'.format(FOLDER, fold), 'r',
                  encoding='utf8') as f:
            for line in f:
                idx, _, y, pred = line.strip().split('\t')
                if MODE == 'pos' and pred == CLASS:
                    indices.append(idx)
                elif MODE == 'truepos' and pred == CLASS and y == pred:
                    indices.append(idx)
                elif MODE == 'falsepos' and pred == CLASS and y != pred:
                    indices.append(idx)

    in_file = '{}/fold-{}/importance_values_{}.txt'.format(FOLDER, fold, CLASS)
    print(in_file)
    with open(in_file, 'r', encoding='utf8') as in_file:
        for i, line in enumerate(in_file):
            idx, feature, score = line.strip().split('\t')
            if MODE != 'all' and idx not in indices:
                continue
            try:
                prev = scores[feature]
            except KeyError:
                prev = []
            scores[feature] = prev + [float(score)]
            if i % 50000 == 0:
                print(i, feature)


print("sorting")
if args.args.combination_method == 'mean':
    glob = [(feature, np.mean(scores[feature]), len(scores[feature]))
            for feature in scores]
else:
    glob = [(feature, np.sqrt(np.sum(np.absolute(scores[feature]))),
             len(scores[feature])) for feature in scores]
glob = sorted(glob, key=lambda x: x[1] if x[2] >= THRESHOLD else -1,
               reverse=True)
print("sorted")


out_file = '{}/importance_values_{}_{}_{}_sorted.tsv'
           .format(FOLDER, args.combination_method, CLASS, MODE)
if args.single_fold:
    out_file = '{}/fold-{}/importance_values_{}_{}_{}_sorted.tsv'
               .format(FOLDER, args.combination_method, args.k, CLASS, MODE)    
print(out_file)
with open(out_file, 'w', encoding='utf8') as out_file:
    out_file.write('FEATURE\t{}\tSUM\tCOUNT\n'
                   .format(args.combination_method.upper()))
    for (feature, score, num) in glob:
        out_file.write('{}\t{:.10f}\t'.format(feature, score))
        entries = scores[feature]
        out_file.write('{:.10f}\t'.format(np.sum(entries)))
        out_file.write('{}\n'.format(num))
