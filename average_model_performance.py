import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

model_dir = args.model
pattern = re.compile('fold-\d+')

acc, f1 = [], []

for _, directories, _ in os.walk(model_dir):
    for fold_dir in directories:
        if not pattern.search(fold_dir):
            continue
        with open('{}/{}/log.txt'.format(model_dir, fold_dir),
                  'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('Accuracy'):
                    acc.append(float(line.strip().split('\t')[1]))
                if line.startswith('F1'):
                    f1.append(float(line.strip().split('\t')[1]))
                    break

try:
    print("Average accuracy", sum(acc) / len(acc))
except ZeroDivisionError:
    print("No accuracy scores")
try:
    print("Average F1 (macro)", sum(f1) / len(f1))
except ZeroDivisionError:
    print("No F1 scores")
with open('{}/model-scores.txt'.format(model_dir), 'w+', encoding='utf8') as f:
    f.write("Average accuracy ({} runs): {}\n"
            "Average macro F1 score ({} runs): {}\n"
            .format(len(acc), sum(acc) / len(acc), len(f1), sum(f1) / len(f1)))
