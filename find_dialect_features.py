from collections import Counter


IN_FILE = 'data/bokmaal+phon_cleaned.tsv'
OUT_FILE = 'data/dialect_features.txt'


ikke = []
jeg, hun, vi, dere, de = [], [], [], [], []
rs, sl = [], []

# question_words = ['hva', 'hvordan', 'hvorfor', 'vem', '']

with open(IN_FILE, 'r', encoding='utf8') as f:
    for line in f:
        utterance = line.strip().split('\t')[4]
        tokens = utterance.split(' ')
        for token in tokens:
            bokmaal, phon = token.split('/')
            if bokmaal == 'ikke':
                ikke.append(phon)
            elif bokmaal == 'jeg':
                jeg.append(phon)
            elif bokmaal == 'hun':
                hun.append(phon)
            elif bokmaal == 'vi':
                vi.append(phon)
            elif bokmaal == 'dere':
                dere.append(phon)
            elif bokmaal == 'de':
                de.append(phon)
            else:
                if 'rs' in bokmaal:
                    if 'rs' in phon and 'ʂ' not in phon:
                        rs.append('rs')
                    elif 'ʂ' in phon and 'rs' not in phon:
                        rs.append('ʂ')
                if 'sl' in bokmaal:
                    if 'sl' in phon and 'ʂl' not in phon:
                        rs.append('sl')
                    elif 'ʂl' in phon and 'sl' not in phon:
                        rs.append('ʂl')


with open(OUT_FILE, 'w+', encoding='utf8') as f:
    f.write('IKKE\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(ikke).most_common()]) + '\n\n')
    f.write('JEG\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(jeg).most_common()]) + '\n\n')
    f.write('HUN\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(hun).most_common()]) + '\n\n')
    f.write('VI\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(vi).most_common()]) + '\n\n')
    f.write('DE\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(de).most_common()]) + '\n\n')
    f.write('DERE\n')
    f.write(', '.join(['{} ({})'.format(*c) for c in Counter(dere).most_common()]) + '\n\n')
