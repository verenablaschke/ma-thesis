import argparse
import numpy as np


def parse_fold(mode, fold_dir, subfolder, label, scores, feature_pfx=''):
    if mode != 'all':
        indices = []
        with open('{}/predictions.tsv'.format(fold_dir), 'r',
                  encoding='utf8') as f:
            for line in f:
                idx, _, y, pred = line.strip().split('\t')
                if mode == 'pos' and pred == label:
                    indices.append(idx)
                elif mode == 'truepos' and pred == label and y == pred:
                    indices.append(idx)
                elif mode == 'falsepos' and pred == label and y != pred:
                    indices.append(idx)

    in_file = '{}/importance_values_{}.txt'.format(subfolder, label)
    print(in_file)
    with open(in_file, 'r', encoding='utf8') as in_file:
        for i, line in enumerate(in_file):
            idx, feature, score = line.strip().split('\t')
            if mode != 'all' and idx not in indices:
                continue
            try:
                prev = scores[feature_pfx + feature]
            except KeyError:
                prev = []
            scores[feature_pfx + feature] = prev + [float(score)]
            if i % 50000 == 0:
                print(i, feature_pfx + feature)
    return scores


def calculate_results(combination_method, scores, min_count, folder,
                      filename_details):
    print("sorting")
    if combination_method == 'mean':
        glob = [(feature, np.mean(scores[feature]), len(scores[feature]))
                for feature in scores]
    else:
        glob = [(feature, np.sqrt(np.sum(np.absolute(scores[feature]))),
                 len(scores[feature])) for feature in scores]
    glob = sorted(glob, key=lambda x: x[1] if x[2] >= min_count else -1,
                  reverse=True)
    print("sorted")

    out_file = '{}/importance_values_{}_sorted.tsv'.format(folder,
                                                           filename_details)
    print(out_file)
    with open(out_file, 'w', encoding='utf8') as out_file:
        out_file.write('FEATURE\t{}\tSUM\tCOUNT\n'
                       .format(combination_method.upper()))
        for (feature, score, num) in glob:
            out_file.write('{}\t{:.10f}\t'.format(feature, score))
            entries = scores[feature]
            out_file.write('{:.10f}\t'.format(np.sum(entries)))
            out_file.write('{}\n'.format(num))


if __name__ == "__main__":
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

    folds = [args.k] if args.single_fold else range(args.k)
    scores = {}
    for fold in folds:
        fold_dir = '{}/fold-{}'.format(args.model, fold)
        parse_fold(args.mode, fold_dir, fold_dir, args.label, scores)

    if args.single_fold:
        filename_details = '{}_{}_{}'.format(args.k, args.label, args.mode)
        folder = '{}/fold-{}'.format(args.model, args.k)
    else:
        filename_details = '{}_{}_{}'.format(args.combination_method,
                                             args.label,
                                             args.mode)
        folder = args.model
    calculate_results(args.combination_method, scores, args.min_count, folder,
                      filename_details)
