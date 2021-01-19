import math

PATH = './models/tweets-'
RUNS = ['w1', 'w2', 'c1', 'c2', 'c3', 'c4', 'c5']
FILENAME = '/features.txt'
MIN_COUNT = 10
MIN_NPMI = 0.3

run2feature2utt = dict()
for run in RUNS:
    feature2utt = dict()
    n_utt = 0  # Identical across runs
    with open(PATH + run + FILENAME, 'r', encoding='utf8') as f:
        for line in f:
            n_utt += 1
            cells = line.strip().split('\t')
            utterance_id = cells[0]
            for feature in cells[1:]:
                try:
                    feature2utt[feature].add(utterance_id)
                except KeyError:
                    feature2utt[feature] = {utterance_id}
    run2feature2utt[run] = feature2utt


# print(run2feature2utt['w1']['<SOS>on<EOS>'])



feature_combo2npmi = dict()
for i in range(len(RUNS) - 1):
    for j in range(i + 1, len(RUNS)):
        print(i, j)
        feature2utt_i = run2feature2utt[RUNS[i]]
        feature2utt_j = run2feature2utt[RUNS[j]]
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


with open('results/features.tsv', 'w+', encoding='utf8') as f:
    for (f_i, f_j), npmi in sorted(feature_combo2npmi.items(), key=lambda item: item[1], reverse=True):
        f.write('{}\t{}\t{:.2f}\n'.format(f_i, f_j, npmi))
