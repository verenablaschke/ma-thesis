from collections import Counter

file1 = 'models/dialects/importance_values_{}_all_sorted_100context.tsv'
file2 = 'models/dialects/importance_values_sqrt_{}_all_sorted_100context.tsv'


def get_features(filename):
    entries = set()
    with open(filename, encoding='utf8') as f:
        next(f)  # Header
        for line in f:
            entries.add(line.split('\t')[1])
    return entries


def summary(filename1, filename2):
    print('1', filename1)
    print('2', filename2)
    entries1 = get_features(filename1)
    entries2 = get_features(filename2)
    intersection = entries1.intersection(entries2)
    print("INTERSECTION", len(intersection))
    print(intersection)
    only1 = entries1.difference(entries2)
    print("\nONLY FILE 1", len(only1))
    print(only1)
    only2 = entries2.difference(entries1)
    print("\nONLY FILE 2", len(only2))
    print(only2)
    print("\n\n")
    return only2


only_sqrt = Counter()
only_sqrt.update(summary(file1.format('nordnorsk'), file2.format('nordnorsk')))
only_sqrt.update(summary(file1.format('vestnorsk'), file2.format('vestnorsk')))
only_sqrt.update(summary(file1.format('troendersk'), file2.format('vestnorsk')))
only_sqrt.update(summary(file1.format('oestnorsk'), file2.format('oestnorsk')))
print("PRESENT IN SQRT(SUM) RESULTS")
for feature, count in only_sqrt.most_common():
    print(feature, count)
