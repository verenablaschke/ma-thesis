import argparse
import numpy as np
import sys


def parse_fold(mode, fold_dir, subfolder, label, combination_method, min_count,
               scale_by_model_score, filename_details, feature_pfx='',
               return_only_scores=False):
    scores = {}
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
            cells = line.strip().split('\t')
            idx, feature, score = cells[0], cells[1], float(cells[2])
            if scale_by_model_score:
                try:
                    model_score = float(cells[3])
                    if model_score < 0:
                        print("MODEL SCORE BELOW ZERO", model_score)
                        model_score = 0.0
                    score *= model_score
                except IndexError:
                    print("NO MODEL SCORE")
                    sys.exit()
            if mode != 'all' and idx not in indices:
                continue
            try:
                prev = scores[feature_pfx + feature]
            except KeyError:
                prev = []
            scores[feature_pfx + feature] = prev + [score]
            if i % 50000 == 0:
                print(i, feature_pfx + feature)
    if return_only_scores:
        return scores
    return calculate_results(combination_method, scores, min_count, subfolder,
                             filename_details)


def global_score(method, scores, feature):
    if method == 'mean':
        return np.mean(scores[feature])
    return np.sqrt(np.sum(np.absolute(scores[feature])))


def calculate_results(combination_method, scores, min_count, folder,
                      filename_details):
    # global value of an importance score (on a fold level)
    print("sorting")
    global_scores_dict = {
        feature: (global_score(combination_method, scores, feature),
                  len(scores[feature]))
        for feature in scores if len(scores[feature]) >= min_count}
    global_scores = [(k, global_scores_dict[k])
                     for k in sorted(global_scores_dict,
                                     key=global_scores_dict.get, reverse=True)]
    print("sorted")

    out_filename = '{}/importance_values_{}_sorted.tsv'.format(
        folder, filename_details)
    print("Writing fold results to", out_filename)
    with open(out_filename, 'w', encoding='utf8') as out_file:
        out_file.write('FEATURE\t{}\tCOUNT\n'
                       .format(combination_method.upper()))
        for (feature, (score, num)) in global_scores:
            out_file.write('{}\t{:.10f}\t{}\n'.format(feature, score, num))
    return global_scores_dict, out_filename


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
    parser.add_argument('--scale', dest='scale_by_model_score',
                        default=False, action='store_true')
    args = parser.parse_args()

    folds = [args.k] if args.single_fold else range(args.k)
    scores_all_folds = []
    filename_details = '{}_{}_{}_{}scaled'.format(
        args.combination_method, args.label, args.mode,
        '' if args.scale_by_model_score else 'un')
    all_features = set()
    for fold in folds:
        fold_dir = '{}/fold-{}'.format(args.model, fold)
        fold_scores = parse_fold(args.mode, fold_dir, fold_dir, args.label,
                                 args.combination_method, args.min_count,
                                 args.scale_by_model_score)[0]
        scores_all_folds.append(fold_scores)
        all_features.update(fold_scores)

    if not args.single_fold:
        print("Averaging scores across folds")
        avg_scores = []  # averaged across folds!
        for feature in all_features:
            avg_scores.append(
                (feature,
                 np.nanmean([fold_scores.get(feature, (np.nan,))[0]
                             for fold_scores in scores_all_folds]),
                 np.nanmean([fold_scores.get(feature, (None, 0))[1]
                             for fold_scores in scores_all_folds])))

        avg_scores = sorted(avg_scores, key=lambda x: x[1], reverse=True)
        print("sorted")

        out_file = '{}/importance_values_{}_sorted.tsv' \
                   .format(args.model, filename_details)
        print("Writing overall results to", out_file)
        with open(out_file, 'w', encoding='utf8') as out_file:
            out_file.write('FEATURE\t{}\tCOUNT\n'
                           .format(args.combination_method.upper()))
            for (feature, score, num) in avg_scores:
                out_file.write('{}\t{:.10f}\t{}\n'.format(feature, score, num))
