import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('importance_file',
                    help='the file created by parse_results.py')
parser.add_argument('correlation_file',
                    help='the file created by feature_correlation.py')
parser.add_argument('--i', destination='min_imp', default=0.03, type=float,
                    help='min. mean importance value')
parser.add_argument('--c', destination='min_corr', default=0.8, type=float,
                    help='min. correlation (NPMI)')
args = parser.parse_args()


importance_values = {}
features = []
with open(args.importance_file, 'r', encoding='utf8') as f:
    next(f)  # Skip header
    for line in f:
        feature, val, _, _ = line.strip().split('\t')
        importance_values[feature] = val
        if val < args.min_imp:
            continue
        features.append(feature)

correlations = {}
with open(args.correlation_file, 'r', encoding='utf8') as f:
    for line in f:
        feature1, feature2, val = line.strip().split('\t')
        if val < args.min_corr:
            continue
        try:
            values = correlations[feature1]
        except KeyError:
            values = {}
        values[feature2] = val
        correlations[feature1] = values


outfile = '.'join(args.importance_file.split('.')[:-1]) + '_correlated.tsv'
with open(outfile, 'r', encoding='utf8') as f:
    for feature in features:
        f.write(feature + '\t' + importance_values[feature])
        try:
            correls = correlations[feature]
            for feat in correls:
                f.write('\t' + feat)
                f.write('\t' + correls[feat])
                try:
                    f.write('\t' + importance_values[feat])
                except KeyError:
                    f.write('\t---')
        except KeyError:
            pass
        f.write('\n')
