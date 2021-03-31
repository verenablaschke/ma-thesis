import argparse
import itertools
import numpy as np
import os
import re
from scipy.spatial.distance import jensenshannon
from parse_results import parse_fold, calculate_results

parser = argparse.ArgumentParser()
parser.add_argument('path_to_fold')
parser.add_argument('type', help='dialects or tweets')
parser.add_argument('--m', dest='mode', help='all/truepos/falsepos/pos',
                    default='all')
parser.add_argument('--comb', dest='combination_method', help='sqrt/sum',
                    default='sqrt')
parser.add_argument('--mc', dest='min_count', default=10, type=int)
parser.add_argument('--n', dest='n_runs', help='number of runs per |Z|',
                    default=10, type=int)
parser.add_argument('--r', dest='n_lime_sample_run', default=None, type=str,
                    help='specific number of LIME samples only')
parser.add_argument('--scale', dest='scale_by_model_score',
                    default=False, action='store_true')
args = parser.parse_args()

fold_dir = args.path_to_fold
labels = '0' if args.type == 'tweets' else ['nordnorsk', 'oestnorsk',
                                            'troendersk', 'vestnorsk']
mode = args.mode
combination_method = args.combination_method
min_count = args.min_count
n_runs = args.n_runs
pattern = re.compile('\d+-\d+')

if args.n_lime_sample_run is not None:
    out_file = fold_dir + '/lime_sample_sizes.tsv'
    with open(out_file, 'w+', encoding='utf8') as f:
        f.write('N_SAMPLES\tAVG_DIST\n')

runs2folders = {}
for _, directories, _ in os.walk(fold_dir):
    for directory in directories:
        if not pattern.search(directory):
            continue
        run = directory.rsplit('-', 1)[0]
        if (args.n_lime_sample_run is not None) and \
                (run != args.n_lime_sample_run):
            continue
        # try:
        #     runs2folders[run].append(directory)
        # except KeyError:
        #     runs2folders[run] = [directory]
        directory = fold_dir + '/' + directory
        scores = {}
        for label in labels:
            filename_details = '{}_{}_{}'.format(label, combination_method,
                                                 mode)
            scores.update(parse_fold(mode, fold_dir, directory, label,
                                     combination_method, min_count,
                                     args.scale_by_model_score,
                                     filename_details, label + '-',
                                     return_only_scores=True))
        filename_details = 'ALL-LABELS_{}_{}'.format(combination_method, mode)
        _, filename = calculate_results(combination_method, scores, min_count,
                                        directory, filename_details)
        try:
            runs2folders[run].append(filename)
        except KeyError:
            runs2folders[run] = [filename]

for run, folder_list in runs2folders.items():
    print('\n')
    print(run)
    print("Getting scores.")
    feature2folder2val = {}
    for folder in folder_list:
        with open(folder,
                  encoding='utf8') as f:
            next(f)  # Header
            for line in f:
                feature, val, _ = line.split('\t')
                val = float(val)
                try:
                    feature2folder2val[feature][folder] = val
                except KeyError:
                    feature2folder2val[feature] = {folder: val}

    print("Getting score distributions")
    array_len = len(feature2folder2val)
    folder2distrib = {folder: np.zeros(array_len, dtype=np.float64)
                      for folder in folder_list}
    for i, feature in enumerate(feature2folder2val):
        for folder in folder_list:
            try:
                folder2distrib[folder][i] = feature2folder2val[feature][folder]
            except KeyError:
                pass

    print("Comparing distributions")
    dist = 0
    count = 0
    for run1, run2 in itertools.combinations(folder_list, 2):
        dist += jensenshannon(folder2distrib[run1], folder2distrib[run2])
        count += 1
    avg_distance = dist / count
    print(avg_distance)

    if args.n_lime_sample_run is not None:
        print("Updating file")
        with open(out_file, 'a', encoding='utf8') as f:
            f.write('{}\t{:.4f}\n'.format(run, avg_distance))
