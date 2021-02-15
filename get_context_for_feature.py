import argparse

parser = argparse.ArgumentParser()
parser.add_argument('feature')
args = parser.parse_args()


with open('data/bokmaal+phon_cleaned.tsv', 'r', encoding='utf8') as f:
    