FREQ_FILE = 'data/nowac/nowac-1.1.lemmas.freq'
OUT_FILE = 'data/top-verbs-fem_subst.txt'
IN_FILE_BOKMAAL = 'data/bokmaal_cleaned.tsv'

verbs = []
fem_nouns = []

get_verbs = True
max_count = 30

min_idx = 1000

with open(FREQ_FILE, encoding='utf8') as f:
    for line in f:
        freq_word, pos = line.strip().split('\t')
        if get_verbs and pos == 'verb':
            _, word = freq_word.split()
            verbs.append(word)
            get_verbs = len(verbs) < max_count
        elif pos == 'subst_fem':
            freq, word = freq_word.split(maxsplit=1)
            if int(freq) <= min_idx:
                break
            if word[-1] == 'e':
                infl = (word[:-1] + 'a', word + 'n')
            else:
                infl = (word + 'a', word + 'en')
            fem_nouns.append((word, infl))

with open(OUT_FILE, 'w+', encoding='utf8') as f:
    f.write('MOST COMMON VERBS\n')
    f.write('\n'.join(verbs))
    f.write('\n\nMOST COMMON FEMININE NOUNS\n')
    f.write('\n'.join([n for n, _ in fem_nouns[:max_count]]))


noun2count = {noun: 0 for noun, _ in fem_nouns}
infl2lemma = {}
for lemma, (infl_a, infl_en) in fem_nouns:
    infl2lemma[infl_a] = lemma
    infl2lemma[infl_en] = lemma

with open(IN_FILE_BOKMAAL, encoding='utf8') as f:
    for line in f:
        utterance = line.strip().split('\t')[4]
        for token in utterance.split(' '):
            try:
                noun2count[infl2lemma[token.lower()]] += 1
            except KeyError:
                pass

counted_nouns = list(noun2count.items())
counted_nouns.sort(key=lambda x: x[1], reverse=True)

with open(OUT_FILE, 'a', encoding='utf8') as f:
    f.write('\n\nMOST COMMON FEMININE NOUNS IN SCANDIASYN\n')
    f.write('\n'.join(['{} ({})'.format(*n) for n in counted_nouns[:50]]))
