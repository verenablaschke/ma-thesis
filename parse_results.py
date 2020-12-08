import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: parse_results.py TYPE CLASS_NR")
    print("Usage: parse_results.py dialects/tweets 0/1/2/3")
    sys.exit()

inf_type = sys.argv[1]
nr = sys.argv[2]
IN_FILE = 'results/results_{}_{}.txt'.format(inf_type, nr)
OUT_FILE = 'results/results_{}_{}_sorted.tsv'.format(inf_type, nr)

THRESHOLD = 10

scores = {}
with open(IN_FILE, 'r', encoding='utf8') as in_file:
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
