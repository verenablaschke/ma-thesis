import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('label', help='the class')
parser.add_argument('mode', help='pos / truepos / falsepos / all')
parser.add_argument('k', help='number of folds', type=int)
parser.add_argument('--min', dest='min_count', help='min. count per fold',
                    default=10, type=int)
args = parser.parse_args()

FOLDER = args.model
CLASS = args.label
THRESHOLD = args.min_count
MODE = args.mode


scores = {}
for fold in range(args.k):
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
means = [(feature,
          np.mean(scores[feature]),
          len(scores[feature])) for feature in scores]
means = sorted(means, key=lambda x: x[1] if x[2] >= THRESHOLD else -1,
               reverse=True)
print("sorted")

with open('{}/importance_values_{}_{}_sorted.tsv'.format(FOLDER, CLASS, MODE),
          'w', encoding='utf8') as out_file:
    out_file.write('FEATURE\tMEAN\tSUM\tCOUNT\n')
    for (feature, mean, num) in means:
        out_file.write('{}\t{:.10f}\t'.format(feature, mean))
        entries = scores[feature]
        out_file.write('{:.10f}\t'.format(np.sum(entries)))
        out_file.write('{}\n'.format(num))
