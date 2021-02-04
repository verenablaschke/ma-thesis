from collections import Counter


IN_FILE = 'data/bokmaal+phon_cleaned.tsv'
OUT_FILE = 'data/dialect_features.txt'


ikke, noe, noen, mye = [], [], [], []
jeg, hun, vi, dere, de = [], [], [], [], []
hva, hvem, hvorfor, når, hvordan, åssen = [], [], [], [], [], []
rs, sl = [], []


word2list = {'ikke': ikke, 'noe': noe, 'noen': noen, 'mye': mye,
             'jeg': jeg, 'hun': hun, 'vi': vi, 'dere': dere, 'de': de,
             'hva': hva, 'hvem': hvem, 'hvorfor': hvorfor, 'når': når,
             'hvordan': hvordan, 'åssen': åssen
             }


with open(IN_FILE, 'r', encoding='utf8') as f:
    for line in f:
        utterance = line.strip().split('\t')[4]
        tokens = utterance.split(' ')
        for token in tokens:
            bokmaal, phon = token.split('/')

            try:
                word2list[bokmaal].append(phon)
            except KeyError:
                pass

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
    for word, variants in word2list.items():
        f.write('{}\n'.format(word.upper()))
        f.write(', '.join(['{} ({})'.format(*c) for c in Counter(variants).most_common()]) + '\n\n')