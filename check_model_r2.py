import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('type', help='dialects or tweets')
parser.add_argument('--k', dest='k', help='number of folds',
                    type=int, default=10)
parser.add_argument('--l', dest='n_lime_feats', type=int, default=100)
args = parser.parse_args()

labels = [0, 1] if args.type == 'tweets' else ['oestnorsk', 'vestnorsk',
                                               'troendersk', 'nordnorsk']

r2_scores = []
for fold in range(args.k):
    for label in labels:
        with open('{}/fold-{}/importance_values_{}.txt'
                  .format(args.model, fold, label), encoding='utf8') as f:
            line_nr = 0
            for line in f:
                if line_nr % args.n_lime_feats == 0:
                    r2_scores.append(float(line.strip().split('\t')[3]))
                line_nr += 1

r2_scores = np.asarray(r2_scores)
print("MAX", np.amax(r2_scores))
print("MIN", np.amin(r2_scores))
print("MEAN", np.mean(r2_scores))
print("STD DEV", np.std(r2_scores))

plt.hist(r2_scores, range=(0.0, 1.0), bins=100)
plt.show()
