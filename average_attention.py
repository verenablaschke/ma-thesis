import argparse
import numpy as np
from collections import Counter
from predict import word2vec_split

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('filename')
parser.add_argument('seq_len', type=int)
parser.add_argument('--f', dest='n_folds', type=int, default=10)
parser.add_argument('--min', dest='min_appearances', type=int, default=20)
parser.add_argument('--d', dest='data_file',
                    default='data/tweets_cleaned_websites.tsv')
parser.add_argument('--s', dest='seq_len', type=int, default=60)
args = parser.parse_args()

token2scores = {}
token2labels = {}

# Attention scores

for i in range(args.n_folds):
    with open('{}/fold-{}/{}'.format(args.model, i, args.filename),
              encoding='utf8') as f:
        next(f)
        for line in f:
            cells = line.strip().split('\t')
            label, pred = cells[0], cells[1]
            scores = cells[2:2 + args.seq_len]
            tokens = cells[2 + args.seq_len:]
            for tok, attn in zip(tokens, scores):
                try:
                    token2scores[tok].append(float(attn))
                    token2labels[tok].append(label + '-' + pred)
                except KeyError:
                    token2scores[tok] = [float(attn)]
                    token2labels[tok] = [label + '-' + pred]

token2avg = [(np.mean(scores), len(scores), tok)
             for tok, scores in token2scores.items()
             if len(scores) >= args.min_appearances]
token2avg.sort(reverse=True)


# Representativeness/distinctiveness

tweets_1 = []
tweets_0 = []
with open(args.data_file, encoding='utf8') as f:
    for line in f:
        _, label, tweet = line.strip().split('\t')
        if label == '1':
            tweets_1.append(tweet)
        else:
            tweets_0.append(tweet)
size_1 = len(tweets_1)
rel_size_1 = len(tweets_1) / (len(tweets_1) + len(tweets_0))
ngrams_1 = word2vec_split(tweets_1, args.seq_len)
feature2count_pos = {}
feature2count_all = {}
for utterance in ngrams_1:
    if len(utterance) < args.seq_len:
        try:
            feature2count_pos['<FILLER>'] += 1
            feature2count_all['<FILLER>'] += 1
        except KeyError:
            feature2count_pos['<FILLER>'] = 1
            feature2count_all['<FILLER>'] = 1
    for ngram in set(utterance):
        try:
            feature2count_pos[ngram] += 1
            feature2count_all[ngram] += 1
        except KeyError:
            feature2count_pos[ngram] = 1
            feature2count_all[ngram] = 1
ngrams_0 = word2vec_split(tweets_0, args.seq_len)
for utterance in ngrams_0:
    if len(utterance) < args.seq_len:
        try:
            feature2count_all['<FILLER>'] += 1
        except KeyError:
            feature2count_all['<FILLER>'] = 1
    for ngram in set(utterance):
        try:
            feature2count_all[ngram] += 1
        except KeyError:
            feature2count_all[ngram] = 1

with open('{}/AVERAGE-{}'.format(args.model,
                                 args.filename.replace('.txt', '.tsv')),
          'w+', encoding='utf8') as f:
    f.write('TOKEN\tAVERAGE_ATTENTION\tREP_POS\tDIST_POS\t'
            'N_OCCURRENCES_TEST\tTRUEPOS\tFALSEPOS\tTRUENEG\tFALSENEG\n')
    for avg, length, token in token2avg:
        labels = Counter(token2labels[token])
        gs_1 = labels['1-1'] + labels['1-0']
        gs_0 = labels['0-0'] + labels['0-1']
        rep_1 = feature2count_pos.get(token, 0) / size_1
        rel_occ = feature2count_pos.get(token, 0) / feature2count_all[token]
        dist_1 = (rel_occ - rel_size_1) / (1 - rel_size_1)
        f.write('{}\t{:.6f}\t{:.6f}\t{:.6f}\t'
                '{}\t{}\t{}\t{}\t{}\n'.format(
                    token, avg, rep_1, dist_1,
                    length,
                    labels['1-1'], labels['0-1'],
                    labels['0-0'], labels['1-0']))
