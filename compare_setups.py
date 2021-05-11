import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('fold')
args = parser.parse_args()

with open(args.fold + '/setup-scores.tsv', 'w+', encoding='utf8') as f_out:
    f_out.write('SET-UP\tF1\tACCURACY\n')
    for _, _, files in os.walk(args.fold):
        for file in files:
            if 'log' in file:
                with open(args.fold + '/' + file, 'r', encoding='utf8') as f_in:
                    for line in f_in:
                        if line.startswith('Accuracy'):
                            acc = line.strip().split('\t')[1]
                        if line.startswith('F1'):
                            f1 = line.strip().split('\t')[1]
                f_out.write('{}\t{}\t{}\n'.format(file, f1, acc))
