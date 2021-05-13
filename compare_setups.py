import argparse
import os
import numpy as np
from pathlib import Path


def get_fold_scores(fold):
    with open(fold + '/setup-scores.tsv', 'w+', encoding='utf8') as f_out:
        f_out.write('SET-UP\tF1\tACCURACY\n')
        for _, _, files in os.walk(fold):
            for file in files:
                if 'log' in file:
                    with open(fold + '/' + file, encoding='utf8') as f_in:
                        for line in f_in:
                            if line.startswith('Accuracy'):
                                acc = line.strip().split('\t')[1]
                            if line.startswith('F1'):
                                f1 = line.strip().split('\t')[1]
                    f_out.write('{}\t{}\t{}\n'.format(file, f1, acc))


def get_aggregate_scores(folds):
    f1_scores = {}
    acc_scores = {}
    for fold in folds:
        with open(fold + '/setup-scores.tsv', encoding='utf8') as f:
            next(f)
            for line in f:
                cells = line.strip().split('\t')
                run, f1, acc = cells[0], float(cells[1]), float(cells[2])
                try:
                    f1_scores[run].append(f1)
                    acc_scores[run].append(acc)
                except KeyError:
                    f1_scores[run] = [f1]
                    acc_scores[run] = [acc]
    f1_list = [(run, np.mean(f1), len(f1)) for run, f1 in f1_scores.items()]
    f1_list.sort(key=lambda x: x[1], reverse=True)
    with open(Path(folds[0]).parent.absolute() / 'setup-scores-aggregate.tsv',
              'w+', encoding='utf8') as f:
        f.write('RUN\tN\tF1\tACCURACY\n')
        for (run, f1_mean, n) in f1_list:
            if n < len(folds) - 1:
                continue
            if run in ('log.txt', 'log_acc.tsv', 'log_f1.tsv'):
                continue
            f.write('{}\t{}\t{:.3f}\t{:.3f}\n'.format(
                run, n, f1_mean, np.mean(acc_scores[run])))


parser = argparse.ArgumentParser()
parser.add_argument('folds', nargs='+')
parser.add_argument('--a', dest='aggregate', default=False,
                    action='store_true')
args = parser.parse_args()

if args.aggregate:
    get_aggregate_scores(args.folds)
else:
    for fold in args.folds:
        get_fold_scores(fold)
