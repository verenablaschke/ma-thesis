from scipy.stats import entropy
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help="path to file")
args = parser.parse_args()

try:
    n_toks = int(args.file.split('-T')[1].split('-')[0])
except ValueError:
    n_toks = int(args.file.split('-T')[1].split('.')[0])
entropy_scores = []

with open(args.file, encoding='utf8') as f:
    next(f)
    for line in f:
        scores = line.split('\t')[2:2 + n_toks]
        entropy_scores.append(entropy([float(x) for x in scores], base=2))

print('mean', np.mean(entropy_scores))
print('min', min(entropy_scores))
print('max', max(entropy_scores))
print('uniform', entropy([1 / n_toks for _ in range(n_toks)]))
print('one-hot', entropy(np.eye(n_toks, 1)))
print('len', len(entropy_scores))
