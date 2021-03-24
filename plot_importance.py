import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to the model')
parser.add_argument('--m', dest='mode', help='pos / truepos / falsepos / all',
                    default='all')
parser.add_argument('--comb', dest='combination_method',
                    help='options: sqrt (square root of sums), mean',
                    default='sqrt', type=str)
parser.add_argument('--scale', dest='scale_by_model_score',
                    default=False, action='store_true')
args = parser.parse_args()

# Produced by feature_context.py
in_file = '{}/importance-spec-rep-{}-{}-{}scaled.tsv' \
                      .format(args.model, args.mode, args.combination_method,
                              '' if args.scale_by_model_score else 'un')
out_file = '{}/figures/importance-{{}}-{}-{}-{}scaled.png' \
                      .format(args.model, args.mode, args.combination_method,
                              '' if args.scale_by_model_score else 'un')
Path('{}/figures/'.format(args.model)).mkdir(parents=True, exist_ok=True)

imp, spec, rep = [], [], []
with open(in_file, 'r', encoding='utf8') as f:
    next(f)  # header
    for line in f:
        cells = line.strip().split('\t')
        imp.append(float(cells[2]))
        rep.append(float(cells[3]))
        spec.append(float(cells[4]))


imp = np.array(imp)
rep = np.array(rep)
spec = np.array(spec)

plt.scatter(imp, rep)
importance_label = "Importance ({}, {}, {}scaled by model error)".format(
    args.combination_method, args.mode,
    '' if args.scale_by_model_score else 'un')
plt.xlabel(importance_label)
plt.ylabel("Representativeness")
plt.savefig(out_file.format('rep'))
plt.show()

plt.scatter(imp, spec)
plt.xlabel(importance_label)
plt.ylabel("Specificity")
plt.savefig(out_file.format('spec'))
plt.show()
