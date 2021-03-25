from collections import Counter
import argparse

parser = argparse.ArgumentParser()
# e.g. 'models/dialects/importance_values_sqrt_{}_all_sorted_100context.tsv'
parser.add_argument('file1')
parser.add_argument('file2')
parser.add_argument('--t', dest='threshold', help='top X features', type=int,
                    default=100)
args = parser.parse_args()


labels = ['nordnorsk', 'vestnorsk', 'oestnorsk', 'troendersk'] \
         if 'dialects' in args.file1 else ['0', '1']


def get_features(filename, threshold):
    entries = set()
    with open(filename, encoding='utf8') as f:
        next(f)  # Header
        i = 0
        for line in f:
            entries.add(line.split('\t')[0])
            i += 1
            if i == threshold:
                break
    return entries


def summary(filename1, filename2, threshold):
    print('1', filename1)
    print('2', filename2)
    entries1 = get_features(filename1, threshold)
    entries2 = get_features(filename2, threshold)
    intersection = entries1.intersection(entries2)
    print("INTERSECTION", len(intersection))
    print(intersection)
    only1 = entries1.difference(entries2)
    print("\nONLY FILE 1:", len(only1))
    print(only1)
    only2 = entries2.difference(entries1)
    print("\nONLY FILE 2:", len(only2))
    print(only2)
    print("\n\n")
    return only1, only2


only_file1, only_file2 = Counter(), Counter()
for label in labels:
    print('#######################\n   {}\n#######################'
          .format(label))
    res = summary(args.file1.format(label), args.file2.format(label),
                  args.threshold)
    only_file1.update(res[0])
    only_file2.update(res[1])

print("ONLY PRESENT IN FILE1 RESULTS", args.file1)
for feature, count in only_file1.most_common():
    print(feature, count)

print("\n\nONLY PRESENT IN FILE2 RESULTS", args.file2)
for feature, count in only_file2.most_common():
    print(feature, count)
