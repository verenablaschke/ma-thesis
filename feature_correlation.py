import math
import argparse
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--c', dest='count', help='minimum count',
                    default='10', type=int)
parser.add_argument('--n', dest='npmi', help='minimum NPMI value',
                    default='0.3', type=float)
args = parser.parse_args()

MIN_COUNT = args.count
MIN_NPMI = args.npmi
args.mode = args.model.strip()

print("Parsing features.")
feature2utt = {}
with open(args.model + '/features.tsv', 'r', encoding='utf8') as f:
    # utterance   label   feature1   feature2   feature3   ...
    header = next(f).strip()
    utt_idx = 0
    for line in f:
        cells = line.strip().split('\t')
        for feature_pkg in cells[2:]:
            for feature in feature_pkg.split():
                try:
                    feature2utt[feature].add(utt_idx)
                except KeyError:
                    feature2utt[feature] = {utt_idx}
        utt_idx += 1

print("Found {} features.".format(len(feature2utt)))
print("Removing rare features")
feature_list = []
for feature, utt in feature2utt.items():
    if len(utt) < MIN_COUNT:
        continue
    feature_list.append((feature, utt))

print("{} features left.".format(len(feature_list)))
print("Calculating co-occurrence scores.")
n_utt = utt_idx
feature_combo2npmi = dict()
for i in range(len(feature_list) - 1):
    if i % 500 == 0:
        print(i, datetime.datetime.now())
    for j in range(i + 1, len(feature_list)):
        f_i, utt_i = feature_list[i]
        f_j, utt_j = feature_list[j]

        utt_inter = len(utt_i.intersection(utt_j))
        if utt_inter == 0:
            # We're only interested in positive correlation.
            continue
        p_xy = utt_inter / n_utt
        p_x = len(utt_i) / n_utt
        p_y = len(utt_j) / n_utt
        h_xy = -math.log(p_xy)
        npmi = math.log(p_xy / (p_x * p_y)) / h_xy
        if npmi < MIN_NPMI:
            continue
        feature_combo2npmi[(f_i, f_j)] = npmi

print("Sorting by NPMI and writing to file.")
with open(args.model + '/features-correlated.tsv', 'w+', encoding='utf8') as f:
    for (f_i, f_j), npmi in sorted(feature_combo2npmi.items(),
                                   key=lambda item: item[1], reverse=True):
        f.write('{}\t{}\t{:.6f}\n'.format(f_i, f_j, npmi))
