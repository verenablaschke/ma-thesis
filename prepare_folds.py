# coding: utf-8

from sklearn import model_selection
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('k', help='k for k-fold cross-validation',
                    default='10', type=int)
args = parser.parse_args()

instances = []
with open(args.model + '/features.tsv', 'r', encoding='utf8') as f:
    next(f)  # Skip header
    for line in f:
        instances.append(line)


kfold = model_selection.KFold(args.k, shuffle=True)

for fold_nr, (train_idx, test_idx) in enumerate(kfold.split(instances)):
    print('Fold #' + str(fold_nr))
    folder = '{}/fold-{}/'.format(args.model, fold_nr)
    Path(folder).mkdir(parents=True, exist_ok=True)
    train_idx.tofile(folder + 'folds-train.npy')
    test_idx.tofile(folder + 'folds-test.npy')
    with open(folder + 'train_data.txt', 'w+', encoding='utf8') as f:
        for idx in train_idx:
            f.write(instances[idx])
    with open(folder + 'test_data.txt', 'w+', encoding='utf8') as f:
        for idx in test_idx:
            f.write(instances[idx])
