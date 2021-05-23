import argparse
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

print("Parsing features.")
feature2labels = {}
labels = []
with open(args.model + '/features.tsv', 'r', encoding='utf8') as f:
    # utterance   label   feature1   feature2   feature3   ...
    header = next(f).strip()
    for line in f:
        cells = line.strip().split('\t')
        label = cells[1]
        labels.append(label)
        features = {feature for feature_pkg in cells[2:]
                    for feature in feature_pkg.split()}
        for feature in features:
            try:
                feature2labels[feature][label] += 1
            except KeyError:
                try:
                    feature2labels[feature][label] = 1
                except KeyError:
                    feature2labels[feature] = {label: 1}

print("Calculating representativeness/distinctiveness.")
n_instances = len(labels)
label_counter = Counter(labels)
labels = list(label_counter.keys())

label2rel_size = {label: label_counter[label] / n_instances for label in labels}

with open(args.model + '/feature-distribution.tsv',
          'w+', encoding='utf8') as f:
    f.write("FEATURE")
    for label in labels:
        f.write('\t{}\t{}-REP\t{}-DIST'.format(label, label, label))
    f.write("\n<ALL>")
    for label in labels:
        f.write("\t{}\t--\t--".format(label_counter[label]))
    f.write("\n")
    for feature, label_counts in feature2labels.items():
        f.write(feature)
        total = sum(label_counts.values())
        for label in labels:
            count = label_counts.get(label, 0)
            rel_occ = count / total
            rel_size = label2rel_size[label]
            dist = (rel_occ - rel_size) / (1 - rel_size)
            f.write("\t{}\t{:.6f}\t{:.6f}".format(count,
                                                  count / label_counter[label],
                                                  dist))
        f.write("\n")

print("Wrote output to " + args.model + '/feature-distribution.tsv')
