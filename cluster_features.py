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
        # print(in_file)
        with open(in_file, 'r', encoding='utf8') as in_file:
            next(in_file)  # Skip header
            for i, line in enumerate(in_file):
                feature, score, _, _ = line.strip().split('\t')
                # if MODE != 'all' and idx not in indices:
                #     continue
                score = float(score)
                try:
                    scores[feature][label][fold] = score
                except KeyError:
                    try:
                        scores[feature][label] = {fold: score}
                    except KeyError:
                        scores[feature] = {label: {fold: score}}

n_features = len(scores)
n_labels = len(labels)
print(n_features, 'features')

example = list(scores)[0]
print(example, scores[example])

feature2idx = {}
label2feature2score = {}
matrix = np.zeros((n_features, n_labels * args.k))
for idx_feat, (feature, label2fold) in enumerate(scores.items()):
    feature2idx[feature] = idx_feat
    for idx_lab, label in enumerate(labels):
        total = 0
        for idx_fold in range(args.k):
            try:
                score = label2fold[label][fold]
                matrix[idx_feat, idx_lab * args.k + idx_fold] = score
                total += score
            except KeyError:
                pass
        try:
            label2feature2score[label][feature] = score
        except KeyError:
            label2feature2score[label] = {feature: score}


pca = PCA(n_components=2)
X = pca.fit_transform(matrix)
print(pca.explained_variance_ratio_)

fig, ax = plt.subplots()

top_features = set()

for label, feature2score in label2feature2score.items():
    top_features.update(f for (f, _) in sorted(feature2score.items(),
        key=lambda x: x[1], reverse=True)[:50])

print(len(top_features), top_features)

# TODO make more flexible
colours = ['red', 'green', 'blue', 'purple']
label2col = {lab: col for (lab, col) in zip(labels, colours)}

for feature in top_features:
    idx = feature2idx[feature]
    top_label = ''
    top_score = -1.0
    for label in labels:
        score_label = label2feature2score[label][feature]
        if score_label > top_score:
            top_label = label
            top_score = score_label
    ax.scatter(X[idx, 0], X[idx, 1], color=label2col[top_label])
    ax.annotate(feature, X[idx])

plt.show()
