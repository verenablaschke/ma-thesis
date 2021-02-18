import math
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import Patch
from matplotlib.pyplot import cm
from scipy.cluster import hierarchy
import matplotlib as mpl
from pathlib import Path

import sys
# avoid RecursionErrors when creating large dendrograms
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes', help='separated by commas, e.g. nordnorsk,vestnorsk,oestnorsk,troendersk')
parser.add_argument('k', help='number of folds', default='10', type=int)
parser.add_argument('numfeatures', help='number of features per class', type=int)
args = parser.parse_args()

print(args)

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



def get_select_features(n=50):
    top_features = set()
    random_features = set()
    for label, feature2score in label2feature2score.items():
        print(label)
        top_features.update(f for (f, _) in sorted(feature2score.items(),
            key=lambda x: x[1], reverse=True)[:n])
        featurelist = list(feature2score.items())
        random.shuffle(featurelist)
        random_features.update(f for (f, _) in featurelist[:n])
    print(len(top_features), len(random_features))
    return top_features, random_features

# TODO make more flexible
colours = ['red', 'green', 'blue', 'purple']
label2col = {lab: col for (lab, col) in zip(labels, colours)}


def get_colour(feature):
    top_label = ''
    top_score = -1.0
    for label in labels:
        score_label = label2feature2score[label][feature]
        if score_label > top_score:
            top_label = label
            top_score = score_label
    return label2col[top_label], top_label


def scatter(X, features, n, selection, add_labels=True):
    fig, ax = plt.subplots()
    for feature in features:
        idx = feature2idx[feature]
        ax.scatter(X[idx, 0], X[idx, 1], color=get_colour(feature)[0])
        if add_labels:
            ax.annotate(feature, X[idx])
    # plt.show()
    fig.set_size_inches(20, 15)
    fig.savefig(args.model + '/figures/scatter-{}-{}.png'.format(n, selection),
                # dpi=200
                # bbox_inches='tight'
                )


def n_to_fontsize(n):
    if n < 20:
        return 10
    if n < 60:
        return 5
    return 2


def n_to_threshold(n):
    # if n < 20:
    #     return 0.7
    return 0.5


def tree(matrix, features, n, selection):
    X_rand = np.zeros((len(features), n_labels * args.k))
    features = list(features)
    for idx, feat in enumerate(features):
        X_rand[idx] = matrix[feature2idx[feat]]
    dist = 1 - cosine_similarity(X_rand)
    Z = linkage(dist, method='average')
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cm.rainbow(np.linspace(0, 1, 10))])
    fig, ax = plt.subplots()
    tree = dendrogram(Z, labels=features, orientation='left',
                      leaf_font_size=n_to_fontsize(n),
                      color_threshold=n_to_threshold(n) * max(Z[:,2]))
    with open(args.model + '/dendrogram-{}-{}.txt'.format(n, selection),
              'w+', encoding='utf8') as f:
        for feature, col in zip(tree['ivl'], tree['color_list']):
            f.write("{}\t{}\t{}\n".format(col, feature, get_colour(feature)[1]))
    features = ax.get_ymajorticklabels()
    for feat in features:
        feat.set_color(get_colour(feat.get_text())[0])
    ax.legend(handles=[Patch(color=col, label=lab) for lab, col in label2col.items()],
              loc='upper left')
    # plt.show()
    fig.set_size_inches(20, 15)
    fig.savefig(args.model + '/figures/dendrogram-{}-{}.png'.format(n, selection),
                # dpi=200
                # bbox_inches='tight'
                )


def create_figs(n, add_labels=True):
    top_features, random_features = get_select_features(n)
    pca = PCA(n_components=2)
    X = pca.fit_transform(matrix)
    print("{:.2f}, {:.2f}".format(*pca.explained_variance_ratio_))
    Path(args.model + '/figures/').mkdir(parents=True, exist_ok=True)
    print("Scatter (top)")
    scatter(X, top_features, n, 'top', add_labels)
    print("Scatter (random)")
    scatter(X, random_features, n, 'random', add_labels)
    print("Dendrogram (top)")
    tree(matrix, top_features, n, 'top')
    print("Dendrogram (random)")
    tree(matrix, random_features, n, 'random')


create_figs(args.numfeatures)
