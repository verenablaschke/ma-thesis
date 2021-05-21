import argparse

parser = argparse.ArgumentParser()
parser.add_argument('importance_file',
                    help='the file created by parse_results.py')
parser.add_argument('correlation_file',
                    help='the file created by feature_correlation.py')
parser.add_argument('--i', dest='min_imp', default=0.03, type=float,
                    help='min. mean importance value')
parser.add_argument('--c', dest='min_corr', default=0.8, type=float,
                    help='min. correlation (NPMI)')
args = parser.parse_args()


print("Reading importance values:", args.importance_file)
importance_values = {}
features = []
with open(args.importance_file, 'r', encoding='utf8') as f:
    next(f)  # Skip header
    for line in f:
        try:
            feature, val, _ = line.strip().split('\t')
        except ValueError:
            continue
        importance_values[feature] = val
        if float(val) < args.min_imp:
            continue
        features.append(feature)

print("Reading correlated features:", args.correlation_file,)
correlations = {}
with open(args.correlation_file, 'r', encoding='utf8') as f:
    for line in f:
        feature1, feature2, val = line.strip().split('\t')
        if float(val) < args.min_corr:
            continue
        try:
            values = correlations[feature1]
        except KeyError:
            values = {}
        values[feature2] = val
        correlations[feature1] = values

outfile = '.'.join(args.importance_file.split('.')[:-1]) + '_correlated.tsv'
print("Writing to file:", outfile)
with open(outfile, 'w+', encoding='utf8') as f:
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
