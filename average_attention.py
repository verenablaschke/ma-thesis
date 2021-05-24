import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('filename')
parser.add_argument('seq_len')
parser.add_argument('--f', dest='n_folds', type=int, default=10)
args = parser.parse_args()

token2scores = {}

for i in range(args.n_folds):
    with open('{}/fold-{}/{}'.format(args.model, i, args.filename),
              encoding='utf8') as f:
        next(f)
        for line in f:
            cells = line.strip().split('\t')
            scores = cells[2:2 + args.seq_len]
            tokens = cells[2 + args.seq_len]
            for tok, attn in zip(tokens, scores):
                try:
                    token2scores[tok].append(float(attn))
                except KeyError:
                    token2scores[tok] = [float(attn)]

token2avg = [(np.mean(scores), tok) for tok, scores in token2scores.items()]
token2avg.sort(reverse=True)

with open('{}/AVERAGE-{}'.format(args.model, args.filename), 'w+',
          encoding='utf8') as f:
    for tok, avg in token2avg:
        f.write('{}\t{}\n'.format(tok, avg))
