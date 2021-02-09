import math
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes', help='separated by commas, e.g. nordnorsk,vestnorsk,oestnorsk,troendersk')
parser.add_argument('k', help='number of folds', default='10', type=int)
args = parser.parse_args()


labels = args.classes.split(',')

# TODO add modes

scores = {}
for fold in range(args.k):
    # if MODE != 'all':
    #     indices = []
    #     with open('{}/fold-{}/predictions.tsv'.format(FOLDER, fold), 'r',
    #               encoding='utf8') as f:
    #         for line in f:
    #             idx, _, y, pred = line.strip().split('\t')
    #             if MODE == 'pos' and pred == CLASS:
    #                 indices.append(idx)
    #             elif MODE == 'truepos' and pred == CLASS and y == pred:
    #                 indices.append(idx)
    #             elif MODE == 'falsepos' and pred == CLASS and y != pred:
    #                 indices.append(idx)
    for label in labels:
        in_file = '{}/fold-{}/importance_values_{}_all_sorted.tsv'.format(args.model, fold, label)
        print(in_file)
        with open(in_file, 'r', encoding='utf8') as in_file:
            next(in_file)  # Skip header
            for i, line in enumerate(in_file):
                feature, score, _, _ = line.strip().split('\t')
                # if MODE != 'all' and idx not in indices:
                #     continue
                try:
                    prev = scores[feature]
                except KeyError:
                    prev = []
                scores[feature] = prev + [float(score)]
                if i % 50000 == 0:
                    print(i, feature)

n_features = len(scores)
print(n_features)
array_len = len(scores[list(scores)[0]])


feature_list = []
matrix = np.zeros((n_features, array_len))
for idx, (feature, array) in enumerate(scores.items()):
    print(feature, array)
    feature_list.append(feature)
    matrix[idx] = np.array(array)


pca = PCA(n_components=2)
X = pca.fit_transform(matrix)
print(pca.explained_variance_ratio_)

fig, ax = plt.subplots()
ax.scatter(X[:50, 0], X[:50, 1])

for i, feature in enumerate(feature_list):
    if i == 50:
        break
    ax.annotate(feature, X[i])

plt.show()
