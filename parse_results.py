import numpy as np

nr = 1
# IN_FILE = 'results/results_{}.txt'.format(nr)
# OUT_FILE = 'results/results_{}_sorted.tsv'.format(nr)
IN_FILE = 'results/results_tweets_{}.txt'.format(nr)
OUT_FILE = 'results/results_tweets_{}_sorted.tsv'.format(nr)


scores = {}
with open(IN_FILE, 'r', encoding='utf8') as in_file:
    for i, line in enumerate(in_file):
        feature, score = line.strip().split('\t')
        try:
            prev = scores[feature]
        except KeyError:
            prev = []
        scores[feature] = prev + [float(score)]
        if i % 50000 == 0:
            print(i, feature)

print("sorting")
# means = [(feature, np.mean(scores[feature])) for feature in scores]
# means = sorted(means, key=lambda x: x[1], reverse=True)
sums = [(feature, np.sum(scores[feature])) for feature in scores]
sums = sorted(sums, key=lambda x: x[1], reverse=True)
print("sorted")

with open(OUT_FILE, 'w', encoding='utf8') as out_file:
    for (feature, score_sum) in sums:
        out_file.write('{}\t{:.10f}\t'.format(feature, score_sum))
        entries = scores[feature]
        out_file.write('{:.10f}\t'.format(np.mean(entries)))
        out_file.write('{}\n'.format(len(entries)))
