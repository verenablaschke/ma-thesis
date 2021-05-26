import argparse
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('file')
args = parser.parse_args()

Path('{}/figures/'.format(args.model)).mkdir(parents=True, exist_ok=True)

attn_scores = []
dist_scores = []
with open(args.file, encoding='utf8') as f:
    next(f)
    for line in f:
        attn, _, dist = line.split('\t')[1:4]
        attn_scores.append(float(attn))
        dist_scores.append(float(dist))

for label in ['0', '1']:
    plt.scatter(attn_scores, dist_scores, color='blue',
                s=3,  # size of dot
                )
plt.axhline(y=0, color='r')
plt.xlabel("Mean attention weight")
plt.ylabel("Distinctiveness")
plt.savefig('{}/figures/attention-distinctiveness.png'.format(args.model))
plt.show()
