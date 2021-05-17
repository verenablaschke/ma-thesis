from collections import Counter


IN_FILE_BOKMAAL_PHONO = 'data/bokmaal+phon_cleaned.tsv'
# IN_FILE_BOKMAAL = 'data/bokmaal_cleaned.tsv'
OUT_FILE = 'data/dialect_features.txt'


ikke, noe, noen, mye = [], [], [], []
jeg, hun, vi, dere, de = [], [], [], [], []
hva, hvem, hvorfor, når, hvordan, åssen = [], [], [], [], [], []

vaere, kunne, skulle, ville, måtte = [], [], [], [], []
gjøre, komme, bruke, skrive, legge = [], [], [], [], []

tid, side, bygd, klokke, klasse = [], [], [], [], []
uke, hytte, helg, elv, søster = [], [], [], [], []
# more ending in -e:
kirke, påske, stue, kone, gate, grense = [], [], [], [], [], []
nouns = ['tida', 'sida', 'bygda', 'klokka', 'klassa',
         'uka', 'hytta', 'helga', 'elva', 'søstera',
         'kirka', 'påska', 'stua', 'kona', 'gata', 'grensa']


rs, sl = [], []

infinitives = []
infinitive_realizations = {}

word2list = {'ikke': ikke, 'noe': noe, 'noen': noen, 'mye': mye,
             'jeg': jeg, 'hun': hun, 'vi': vi, 'dere': dere, 'de': de,
             'hva': hva, 'hvem': hvem, 'hvorfor': hvorfor, 'når': når,
             'hvordan': hvordan, 'åssen': åssen,
             'være': vaere, 'kunne': kunne, 'skulle': skulle,
             'ville': ville, 'måtte': måtte,
             'gjøre': gjøre, 'komme': komme, 'bruke': bruke,
             'skrive': skrive, 'legge': legge,
             }

noun2list = {'tida': tid, 'sida': side, 'bygda': bygd, 'klokka': klokke,
             'klassa': klasse, 'uka': uke, 'hytta': hytte, 'helga': helg,
             'elva': elv, 'søstera': søster, 'kirka': kirke,
             'påska': påske, 'stua': stue, 'kona': kone, 'gata': gate,
             'grensa': grense,
             'tiden': tid, 'siden': side, 'bygden': bygd, 'klokken': klokke,
             'klassen': klasse, 'uken': uke, 'hytten': hytte, 'helgen': helg,
             'elven': elv, 'søsteren': søster, 'kirken': kirke,
             'påsken': påske, 'stuen': stue, 'konen': kone, 'gaten': gate,
             'grensen': grense}

with open(IN_FILE_BOKMAAL_PHONO, 'r', encoding='utf8') as f:
    for line in f:
        utterance = line.strip().split('\t')[4]
        tokens = utterance.split(' ')
        infinitive = False
        for idx, token in enumerate(tokens):
            bokmaal, phon = token.split('/')
            bokmaal = bokmaal.lower()
            try:
                word2list[bokmaal].append(phon)
            except KeyError:
                pass
            try:
                noun2list[bokmaal].append(phon)
            except KeyError:
                pass
            if 'rs' in bokmaal:
                if 'rs' in phon and 'ʂ' not in phon:
                    rs.append('rs')
                elif 'ʂ' in phon and 'rs' not in phon:
                    rs.append('ʂ')
            if 'sl' in bokmaal:
                if 'sl' in phon and 'ʂl' not in phon:
                    sl.append('sl')
                elif 'ʂl' in phon and 'sl' not in phon:
                    sl.append('ʂl')
            if infinitive:
                infinitives.append(bokmaal)
                try:
                    infinitive_realizations[bokmaal].append(phon)
                except KeyError:
                    infinitive_realizations[bokmaal] = [phon]
            infinitive = bokmaal == 'å'


word2list['rs'] = rs
word2list['sl'] = sl


with open(OUT_FILE, 'w+', encoding='utf8') as f:
    for word, variants in word2list.items():
        f.write('{}\n'.format(word.upper()))
        f.write(', '.join(['{} ({})'.format(*c)
                           for c in Counter(variants).most_common()]) + '\n\n')
    for noun in nouns:
        f.write('{}\n'.format(noun.upper()))
        f.write(', '.join(['{} ({})'.format(*c)
                           for c in Counter(noun2list[noun]).most_common()]) + '\n\n')
    f.write('INFINITIVES\n')
    for entry, count in Counter(infinitives).most_common(30):
        f.write('{} ({}): '.format(entry, count))
        f.write(', '.join(['{} ({})'.format(*c)
                           for c in Counter(
                            infinitive_realizations[entry]).most_common(5)]))
        f.write('\n')
