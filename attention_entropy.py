from scipy.stats import entropy
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('filename')
parser.add_argument('--f', dest='n_folds', type=int, default=10)
args = parser.parse_args()

try:
    n_toks = int(args.filename.split('-T')[1].split('-')[0])
except ValueError:
    n_toks = int(args.filename.split('-T')[1].split('.')[0])

entropy_scores = []
for i in range(args.n_folds):
    with open('{}/fold-{}/{}'.format(args.model, i, args.filename),
              encoding='utf8') as f:
        next(f)
        for line in f:
            scores = line.split('\t')[2:2 + n_toks]
            entropy_scores.append(entropy([float(x) for x in scores]))

with open('{}/ENTROPY-{}'.format(args.model, args.filename), 'w+', encoding='utf8') as f:
    f.write('Mean\t{:.2f}\n'.format(np.mean(entropy_scores)))
    f.write('Std dev\t{:.2f}\n'.format(np.std(entropy_scores)))
    f.write('Min\t{:.2f}\n'.format(np.min(entropy_scores)))
    f.write('Max\t{:.2f}\n'.format(np.max(entropy_scores)))
    f.write('Uniform\t{:.2f}\n'.format(entropy([1 / n_toks for _ in range(n_toks)])))
    f.write('One-hot\t{:.2f}\n'.format(entropy(np.eye(n_toks, 1))[0]))
    f.write('Len\t{:.2f}\n'.format(len(entropy_scores)))

print('mean', np.mean(entropy_scores))
print('std dev', np.std(entropy_scores))
print('min', min(entropy_scores))
print('max', max(entropy_scores))
print('uniform', entropy([1 / n_toks for _ in range(n_toks)]))
print('one-hot', entropy(np.eye(n_toks, 1))[0])
print('len', len(entropy_scores))
