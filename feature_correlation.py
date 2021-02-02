import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--c', dest='count', help='minimum count',
                    default='10', type=int)
parser.add_argument('--n', dest='npmi', help='minimum NPMI value',
                    default='0.3', type=float)
args = parser.parse_args()

MIN_COUNT = args.count
MIN_NPMI = args.npmi


lvl2feature2utt = dict()
with open(args.model + '/features.tsv', 'r', encoding='utf8') as f:
    # utterance   label   feature1   feature2   feature3   ...
    header = next(f).strip()
    feature_lvls = header.split('\t')[2:]
    utt_idx = 0
    for line in f:
        cells = line.strip().split('\t')
        for lvl, feature_pkg in zip(feature_lvls, cells[2:]):
            try:
                feature2utt = lvl2feature2utt[lvl]
            except KeyError:
                feature2utt = dict()
            for feature in feature_pkg.split():
                try:
                    feature2utt[feature].add(utt_idx)
                except KeyError:
                    feature2utt[feature] = {utt_idx}
            lvl2feature2utt[lvl] = feature2utt
        utt_idx += 1

# print(lvl2feature2utt['word-1']['<SOS>on<EOS>'])


n_utt = utt_idx - 1
n_lvls = len(feature_lvls)
feature_combo2npmi = dict()
for i in range(n_lvls - 1):
    for j in range(i + 1, n_lvls):
        print(i, j)
        feature2utt_i = lvl2feature2utt[feature_lvls[i]]
        feature2utt_j = lvl2feature2utt[feature_lvls[j]]
        for f_i, utt_i in feature2utt_i.items():
            if len(utt_i) < MIN_COUNT:
                continue
            del_list = []
            for f_j, utt_j in feature2utt_j.items():
                if len(utt_j) < MIN_COUNT:
                    del_list.append(f_j)
                    continue
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

        # Remove rare features so they are automatically skipped in future loops
        for key in del_list:
            del feature2utt_j[key]    


with open(args.model + '/features-correlated.tsv', 'w+', encoding='utf8') as f:
    for (f_i, f_j), npmi in sorted(feature_combo2npmi.items(),
                                   key=lambda item: item[1], reverse=True):
        f.write('{}\t{}\t{:.2f}\n'.format(f_i, f_j, npmi))
