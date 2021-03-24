import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('in_file',
                    help='path-to-model/importance-spec-rep-[]-.tsv')
args = parser.parse_args()

imp, spec, rep = [], [], []
with open(args.in_file, 'r', encoding='utf8') as f:
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
plt.show()

plt.scatter(imp, spec)
plt.show()
