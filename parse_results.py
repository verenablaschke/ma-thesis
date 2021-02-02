import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: parse_results.py MODEL FOLD_COUNT CLASS")
    print("E.g.: parse_results.py models/dialects10 10 nordnorsk")
    sys.exit()

folder = sys.argv[1]
label = sys.argv[3]
THRESHOLD = 10  # per fold
OUT_FILE = '{}/importance_values_{}_sorted.tsv'.format(folder, label)
scores = {}


for fold in range(int(sys.argv[2])):
    in_file = '{}/fold-{}/importance_values_{}.txt'.format(folder, fold, label)
    print(in_file)
    with open(in_file, 'r', encoding='utf8') as in_file:
        for i, line in enumerate(in_file):
            _, feature, score = line.strip().split('\t')
            try:
                prev = scores[feature]
            except KeyError:
                prev = []
            scores[feature] = prev + [float(score)]
            if i % 50000 == 0:
                print(i, feature)

print("sorting")
means = [(feature, np.mean(scores[feature]), len(scores[feature])) for feature in scores]
means = sorted(means, key=lambda x: x[1] if x[2] >= THRESHOLD else -1, reverse=True)
# sums = [(feature, np.sum(scores[feature])) for feature in scores]
# sums = sorted(sums, key=lambda x: x[1], reverse=True)
print("sorted")

with open(OUT_FILE, 'w', encoding='utf8') as out_file:
    for (feature, mean, num) in means:
        out_file.write('{}\t{:.10f}\t'.format(feature, mean))
        entries = scores[feature]
        out_file.write('{:.10f}\t'.format(np.sum(entries)))
        # out_file.write('{}\n'.format(len(entries)))
        out_file.write('{}\n'.format(num))
