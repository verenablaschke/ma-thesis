# coding=utf-8

import os
import re

# 'PHON', 'ORTHO', 'BOTH'
MODE = 'BOTH'
MIN_WORDS_PER_UTTERANCE = 3

DATA_DIR = './data/'
if MODE == 'PHON':
    INPUT_DIR = DATA_DIR + 'ndc_phon_with_informant_codes/files/'
    OUT_FILE = DATA_DIR + 'phon_cleaned.tsv'
    LOG_FILE = DATA_DIR + 'phon_log.txt'
elif MODE == 'ORTHO':
    INPUT_DIR = DATA_DIR + 'ndc_with_informant_codes/files/'
    OUT_FILE = DATA_DIR + 'bokmaal_cleaned.tsv'
    LOG_FILE = DATA_DIR + 'bokmaal_log.txt'
else:
    INPUT_DIR = DATA_DIR + 'ndc_phon_with_informant_codes/files/'
    INPUT_DIR_ORTHO = DATA_DIR + 'ndc_with_informant_codes/files/'
    OUT_FILE = DATA_DIR + 'bokmaal+phon_cleaned.tsv'
    LOG_FILE = DATA_DIR + 'bokmaal+phon_log.txt'


# Counties in the dataset (pre-reform)

nordnorsk = {'finnmark': ['hammerfest', 'kautokeino', 'kirkenes',
                          'kjoellefjord', 'lakselv', 'tana', 'vardoe'],
             'nordland': ['ballangen', 'beiarn', 'bodoe', 'hattfjelldal',
                          'heroeyN', 'mo_i_rana', 'myre', 'narvik',
                          'stamsund', 'steigen', 'soemna'],
             'troms': ['botnhamn', 'karlsoey', 'kirkesdalen', 'kvaefjord',
                       'kvaenangen', 'kaafjord', 'lavangen', 'medby',
                       'mefjordvaer', 'stonglandseidet', 'tromsoe']}
# Indre Troms: Kirkesdalen
# Northern Sami: Kåfjord

troendersk = {'nord_troendelag': ['inderoey', 'lierne', 'meraaker',
                                  'namdalen'],
              'soer_troendelag': ['bjugn', 'gauldal', 'oppdal', 'roeros',
                                  'selbu', 'skaugdalen', 'stokkoeya',
                                  'trondheim'],
              'moere_og_romsdal': ['aure', 'surnadal', 'todalen']}

vestnorsk = {'hordaland': ['bergen', 'boemlo', 'eidfjord', 'fusa',
                           'kvinnherad', 'lindaas', 'voss'],
             'moere_og_romsdal': ['bud', 'heroeyMR', 'rauma', 'stranda',
                                  'volda'],
             'rogaland': ['gjesdal', 'hjelmeland', 'karmoey', 'sokndal',
                          'stavanger', 'suldal', 'time'],
             'sogn_og_fjordane': ['hyllestad', 'joelster', 'kalvaag',
                                  'luster', 'stryn'],
             'aust_agder': ['evje', 'landvik', 'valle', 'vegaarshei'],
             'vest_agder': ['kristiansand', 'lyngdal', 'sirdal', 'vennesla',
                            'aaseral']}
oestnorsk = {'akershus': ['enebakk', 'lommedalen', 'nes'],
             'buskerud': ['darbu', 'flaa', 'rollag', 'sylling', 'aal'],
             'hedmark': ['alvdal', 'dalsbygda', 'drevsjoe', 'kirkenaer',
                         'rena', 'stange', 'trysil'],
             'oppland': ['brekkom', 'gausdal', 'jevnaker', 'kvam', 'lom',
                         'skreia', 'vang', 'vestre_slidre'],
             'telemark': ['hjartdal', 'langesund', 'nissedal', 'tinn',
                          'vinje'],
             'vestfold': ['brunlanes', 'hof', 'lardal'],
             'oestfold': ['aremark', 'fredrikstad', 'roemskog']}

# The dialect area division is based on
# Mæhlum & Røyneland 2018: Det norske dialektlandskapet
norwegian = {'nordnorsk': nordnorsk,
             'troendersk': troendersk,
             'vestnorsk': vestnorsk,
             'oestnorsk': oestnorsk}
norwegian_places = []
place2county = {}
place2area = {}
for dialect_area in norwegian:
    for county in norwegian[dialect_area]:
        for place in norwegian[dialect_area][county]:
            norwegian_places.append(place)
            place2area[place] = dialect_area
            place2county[place] = county


skip_tokens = ['#', '##',  # pauses
               '*',  # overlapping utterances
               '?', '!', '"', '...', '…', '"',
               # "Interjeksjonar vi ikkje endrar stavemåten på"
               'ee', 'eh', 'ehe', 'em', 'heh', 'hm', 'm', 'm-m', 'mhm', 'mm'
               ]
if MODE == 'ORTHO' or MODE == 'BOTH':
    skip_tokens.append('e')


class Line_Iter(object):

    def __init__(self, in_file):
        self.iterator = iter(in_file)
        self.saved = []

    def __iter__(self):
        return self.iterator

    def next(self):
        if self.saved:
            return self.saved.pop(0)
        return next(self.iterator)

    def next_iter_only(self):
        return next(self.iterator)

    def save_line(self, line):
        self.saved.append(line)


def is_named_entity(string):
    # Names: F1 (F2, F3, ...), M1, E1
    # Other NEs: N1
    # Names of interview participants use the participant code,
    # which contains a number.
    return bool(re.search(r'\d', string))


phono_replace_first = {'nng': 'ŋŋ', 'ng': 'ŋ', 'kkj': 'çç',
                       'ssj': 'ʂʂ', 'ttj': 'tʧ'}
phono_replace_next = {'\'ŋ': 'ŋ̍', '\'m': 'm̩', '\'n': 'n̩', '\'l': 'l̩',
                      '\'r': 'r̩', '\'s': 's̩', '\'L': ' ̍ɽ', 'L': 'ɽ',
                      'kj': 'ç', 'sj': 'ʂ', 'tj': 'ʧ',
                      '\'i': '.i', '\'e': '.e', '\'u': '.u', '\'æ': '.æ',
                      '\'o': '.o', '\'y': '.y', '\'a': '.a'}


def clean_phono(utterance):
    for k, v in phono_replace_first.items():
        utterance = utterance.replace(k, v)
    for k, v in phono_replace_next.items():
        utterance = utterance.replace(k, v)
    return utterance


places = set()
informants = set()
area2towns, area2files = {}, {}
skipped = []
_, _, filenames = next(os.walk(INPUT_DIR))
with open(OUT_FILE, 'w', encoding='utf8') as out_file:
    for file in filenames:
        place = file.split('_')[0]
        if place not in norwegian_places:
            continue
        places.add(place)
        if MODE == 'BOTH':
            try:
                in_file_ortho = open(INPUT_DIR_ORTHO + file, 'r',
                                     encoding='utf8')
                line_iter_ortho = Line_Iter(in_file_ortho)
            except FileNotFoundError:
                print('The file \'' + file + '\' does not have an orthography-'
                      'based counterpart. (Skipping file.)')
                skipped.append(file)
                continue
        with open(INPUT_DIR + file, 'r', encoding='utf8') as in_file:
            line_iter = Line_Iter(in_file)
            try:
                while True:
                    line = line_iter.next()
                    if MODE == 'BOTH':
                        # Some lines are empty except for the informant code.
                        # Remove these lines to make sure the phonetic and
                        # orthographic versions are still lined up properly.
                        tokens_phon = line.strip().split()
                        while len(tokens_phon) <= 1:
                            tokens_phon = line_iter.next().strip().split()
                        speaker = tokens_phon[0]
                        tokens_ortho = []
                        try:
                            while len(tokens_ortho) <= 1:
                                line_ortho = line_iter_ortho.next().replace(
                                    '"', '')
                                tokens_ortho = line_ortho.strip().split()
                        except StopIteration:
                            # Sometimes, the ortho file is missing the last
                            # few line(s).
                            break
                        # Sometimes, a longer utterance or sequence of
                        # utterances is encoded as a single utterance in the
                        # ortho file, but as a sequence of utterances by the
                        # same speaker in the phono file.
                        # -> Split the ortho utterance into shorter parts.
                        n_toks_total = len(tokens_phon)
                        if len(tokens_ortho) > n_toks_total:
                            cur_tokens_ortho = tokens_ortho[:n_toks_total]
                            n_toks_ortho = len(tokens_ortho)
                            while n_toks_ortho > n_toks_total:
                                next_line = line_iter.next_iter_only().strip()
                                line_iter.save_line(next_line)
                                toks = next_line.split()
                                # Skip speaker code (token 0)
                                n_toks = len(toks[1:])
                                next_line_ortho = ' '.join(tokens_ortho[
                                    n_toks_total:n_toks_total + n_toks])
                                line_iter_ortho.save_line(
                                    speaker + ' ' + next_line_ortho)
                                n_toks_total += n_toks
                            if n_toks_ortho != n_toks_total:
                                print("DIFF LENGTHS",
                                      n_toks_ortho, n_toks_total)
                            tokens_ortho = cur_tokens_ortho

                    else:
                        tokens = line.strip().split()
                        speaker = tokens[0]

                    if not speaker.startswith(place):
                        # Interviewer, not informant
                        continue
                    informants.add(speaker)
                    utterance = []

                    if MODE == 'BOTH':
                        if len(tokens_phon[1:]) != len(tokens_ortho[1:]):
                            print('DIFFERENT UTTERANCE LENGTHS', file)
                            print(tokens_phon[1:])
                            print(tokens_ortho[1:])
                        for tok_phon, tok_ortho in zip(tokens_phon[1:],
                                                       tokens_ortho[1:]):
                            if tok_phon.endswith('-'):
                                continue
                            if tok_ortho in skip_tokens:
                                continue
                            if is_named_entity(tok_phon):
                                continue
                            if '/' in tok_ortho or '/' in tok_phon:
                                print('MAL-FORMED TOKEN?', tok_ortho, tok_phon)
                                tok_ortho = tok_ortho.replace('/', '')
                                tok_phon = tok_phon.replace('/', '')
                            tok_phon = tok_phon.replace('_', '')
                            tok_ortho = tok_ortho.replace('_', '')
                            if len(tok_phon) > 0 and tok_phon[0].islower():
                                # Capitalization check to exclude place names
                                utterance.append(tok_ortho + '/' + clean_phono(
                                    tok_phon))
                    else:
                        for token in tokens[1:]:
                            if token.endswith('-'):
                                continue
                            if token in skip_tokens:
                                continue
                            if is_named_entity(token):
                                continue
                            token = token.replace('_', '')
                            if len(token) > 0 and token[0].islower():
                                # Capitalization check to exclude place names
                                utterance.append(clean_phono(token))
                    if len(utterance) < MIN_WORDS_PER_UTTERANCE:
                        continue
                    utterance = ' '.join(utterance).strip()
                    out_file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                        place2area[place], place2county[place],
                        place, file, utterance))
                    try:
                        area2towns[place2area[place]].add(place)
                    except KeyError:
                        area2towns[place2area[place]] = {place}
                    try:
                        area2files[place2area[place]].append(file)
                    except KeyError:
                        area2files[place2area[place]] = [file]
            except StopIteration:
                pass

        if MODE == 'BOTH':
            in_file_ortho.close()


gk, uk, gm, um, other = [], [], [], [], []
for informant in informants:
    code = informant[-2:]
    if code == 'gk':
        gk.append(informant)
    elif code == 'uk':
        uk.append(informant)
    elif code == 'gm':
        gm.append(informant)
    elif code == 'um':
        um.append(informant)
    else:
        other.append(informant)

with open(LOG_FILE, 'w', encoding='utf8') as log_file:
    log_file.write('No. of places: ' + str(len(places)) + '\n')
    log_file.write(str(places) + '\n\n')
    for area, towns in area2towns.items():
        log_file.write('{}: {} / {}\n'.format(area, len(towns),
                                              len(area2files[area])))
        log_file.write(str(towns) + '\n\n')
        # log_file.write(str(area2files[area]) + '\n\n')
    log_file.write('No. of informants: ' + str(len(informants)) + '\n')
    log_file.write('GAMMEL KVINNE: ' + str(len(gk)) + '\n')
    log_file.write(str(gk) + '\n\n')
    log_file.write('UNG KVINNE: ' + str(len(uk)) + '\n')
    log_file.write(str(uk) + '\n\n')
    log_file.write('GAMMEL MANN: ' + str(len(gm)) + '\n')
    log_file.write(str(gm) + '\n\n')
    log_file.write('UNG MANN: ' + str(len(um)) + '\n')
    log_file.write(str(um) + '\n\n')
    log_file.write('OTHER: ' + str(len(other)) + '\n')
    log_file.write(str(other) + '\n\n')
    if MODE == 'BOTH':
        log_file.write('Skipped (no orthographic counterpart):\n')
        log_file.write(str(skipped) + '\n\n')
