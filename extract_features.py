# coding: utf-8

import pandas as pd
from pathlib import Path
import re
import sys
import argparse
from collections import Counter
from transformers import FlaubertTokenizer, BertTokenizer

ESCAPE_TOKS = ['<URL>', '<USERNAME>', '<HASHTAG>', '<NUMBER>']


def get_escape_toks():
    return ESCAPE_TOKS


def add_ngram(featuremap, label, ngram, context):
    try:
        featuremap[label][ngram].append(context)
    except KeyError:
        try:
            featuremap[label][ngram] = [context]
        except KeyError:
            featuremap[label] = {ngram: [context]}
    return featuremap


def update_featuremap(featuremap, label, ngram, context):
    if featuremap is not None:
        featuremap = add_ngram(featuremap, label, ngram, context)
        featuremap = add_ngram(featuremap, 'all', ngram, context)
    return featuremap


# U0329, U030D are the combining lines for marking syllabic consonants
char_pattern = re.compile(r'(\w[\u0329\u030D]*|\.\w)',
                          re.UNICODE | re.IGNORECASE)


def utterance2ngrams(utterance, label, outfile, word_ns, char_ns,
                     featuremap, verbose=False,
                     utterance_lvl_featuremap=True):
    utterance = utterance.strip().replace('\n', ' ')
    if DIALECTS:
        words = utterance.split(' ')
        words_wordlvl = []
        words_charlvl = []
        for word in words:
            try:
                words_wordlvl.append(word)
                # Phonetic string only:
                words_charlvl.append(word.split('/')[1])
            except ValueError:
                print('Word does not contain \'/\':', word)
                print('utterance:', utterance)
                sys.exit()
    else:
        # utterance = utterance.lower()
        utterance = utterance.replace(',', '')
        utterance = utterance.replace(';', '')
        utterance = utterance.replace('.', '')
        utterance = utterance.replace('’', "'")
        words_wordlvl = utterance.split(' ')
        words_charlvl = words_wordlvl

    ngrams = []
    sep = '<SEP>'
    for word_n in word_ns:
        cur_ngrams = []
        for i in range(len(words_wordlvl) + 1 - word_n):
            tokens = words_wordlvl[i:i + word_n]
            if not DIALECTS:
                tokens = [tok if tok in ESCAPE_TOKS else tok.lower()
                          for tok in tokens]
            ngram = sep.join(tokens)
            # Padding to distinguish these from char n-grams
            if args.add_uncased:
                only_escaped = True
                tmp_tokens = []
                for tok in tokens:
                    if tok in ESCAPE_TOKS:
                        tmp_tokens.append(tok)
                    else:
                        only_escaped = False
                        tmp_tokens.append(tok.lower())
                if args.uncased_only:
                    uncased_ngram = '<SOS>{}<EOS>'.format(sep.join(tmp_tokens))
                    cur_ngrams.append(uncased_ngram)
                    featuremap = update_featuremap(featuremap, label, uncased_ngram,
                                      '<SOS>' + ngram + '<EOS>')
                else:
                    if not only_escaped:
                        uncased_ngram = 'UNCASED:<SOS>{}<EOS>'.format(
                            sep.join(tmp_tokens))
                        cur_ngrams.append(uncased_ngram)
                        featuremap = update_featuremap(featuremap, label, uncased_ngram,
                                          '<SOS>' + ngram + '<EOS>')
            if not args.uncased_only:
                ngram = '<SOS>' + ngram + '<EOS>'
                cur_ngrams.append(ngram)
        ngrams.append(cur_ngrams)
    for char_n in char_ns:
        cur_ngrams = []
        for word, context in zip(words_charlvl, words_wordlvl):
            if word in ESCAPE_TOKS:
                continue
            chars = list(char_pattern.findall(word))
            word_len = len(chars)
            if word_len == 0:
                continue
            if char_n > 1:
                pfx = chars[:char_n - 1]
                ngram = sep + ''.join(pfx)
                cur_ngrams.append(ngram)
                featuremap = update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    if args.uncased_only:
                        uncased_ngram = ngram.lower()
                    else:
                        uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    featuremap = update_featuremap(featuremap, label,
                                      uncased_ngram, context)
            for i in range(len(chars) + 1 - char_n):
                ngram = ''.join(chars[i:i + char_n])
                cur_ngrams.append(ngram)
                featuremap = update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    if args.uncased_only:
                        uncased_ngram = ngram.lower()
                    else:
                        uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    featuremap = update_featuremap(featuremap, label,
                                      uncased_ngram, context)

            if char_n > 1:
                sfx = chars[word_len + 1 - char_n:]
                ngram = ''.join(sfx) + sep
                cur_ngrams.append(ngram)
                featuremap = update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    if args.uncased_only:
                        uncased_ngram = ngram.lower()
                    else:
                        uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    featuremap = update_featuremap(featuremap, label,
                                      uncased_ngram, context)
        ngrams.append(cur_ngrams)
    if verbose:
        print(utterance)
        for lvl_ngrams in ngrams:
            print(lvl_ngrams)

    if not outfile:
        return None, featuremap

    ngrams_flat = []
    with open(outfile, 'a', encoding='utf8') as f:
        f.write(utterance + '\t' + str(label))
        for lvl_ngrams in ngrams:
            lvl_ngrams = list(set(lvl_ngrams))
            ngrams_flat += lvl_ngrams
            f.write('\t')
            f.write(' '.join(lvl_ngrams))
        f.write('\n')
    return ngrams_flat, featuremap


def utterance2bpe_toks(tokenizer, utterance, label, outfile,
                       featuremap, verbose=False):
    toks = []
    initial_tokens = re.split("([^\w<>]+)", utterance)
    n_initial_toks = len(initial_tokens)
    skip = False
    for idx, token in enumerate(initial_tokens):
        if skip:
            skip = False
            continue
        token = token.strip()
        if not token:
            continue
        if idx + 1 < n_initial_toks and initial_tokens[idx + 1] == "'":
            token += "'"
            skip = True
        if token in ESCAPE_TOKS:
            subtokens = [token]
        else:
            subtokens = [t for t in tokenizer.bpe(token).split(" ")]
        for subtok in subtokens:
            featuremap = update_featuremap(featuremap, label, subtok, token)
        toks.extend(subtokens)
    if verbose:
        print(utterance)
        print(toks)
    if outfile:
        with open(outfile, 'a', encoding='utf8') as f:
            f.write("{}\t{}\t{}\n".format(utterance, label, ' '.join(toks)))
    return toks, featuremap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=['dialects', 'tweets'])
    parser.add_argument('model')
    parser.add_argument('--word', dest='word_ngrams', default='[1,2]', type=str)
    parser.add_argument('--char', dest='char_ngrams', default='[2,3,4,5]',
                        type=str)
    parser.add_argument('--perc', dest='percentage', default=False,
                        action='store_true',
                        help='use ratios in the featuremap file')
    parser.add_argument('--lower', dest='add_uncased', default=False,
                        action='store_true')
    parser.add_argument('--loweronly', dest='uncased_only', default=False,
                        action='store_true')
    parser.add_argument('--i', dest='input_file', default=None,
                        help='path to the input file if not '
                             'data/bokmaal+phon_cleaned.tsv or '
                             'data/tweets_cleaned.tsv')
    parser.add_argument('--bpe', dest='bpe', default=False,
                        help='BPE tokenization (for Flaubert embeddings)',
                        action='store_true')
    parser.add_argument('--embmod', dest='embedding_model',
                        default='flaubert/flaubert_base_cased',
                        choices=['flaubert/flaubert_base_cased',
                                 'flaubert/flaubert_small_cased',
                                 'flaubert/flaubert_large_cased',
                                 'bert-base-multilingual-cased'],
                        type=str)
    # python3 extract_features.py tweets '' --utt "test utterance here"
    parser.add_argument('--utt', dest='single_utterance', default=None, type=str)
    args = parser.parse_args()

    if args.uncased_only:
        args.add_uncased = True

    DIALECTS = args.type == 'dialects'

    args.word_ngrams = args.word_ngrams.strip()
    args.char_ngrams = args.char_ngrams.strip()
    if args.word_ngrams[0] != '[' or args.word_ngrams[-1] != ']' \
            or args.char_ngrams[0] != '[' or args.char_ngrams[-1] != ']':
        print('The list of n-gram levels needs to be enclosed in square brackets,'
              ' e.g. [1,2,3] or [].')
        sys.exit()
    WORD_NS = args.word_ngrams[1:-1]
    if len(WORD_NS) == 0:
        WORD_NS = []
    else:
        WORD_NS = [int(i) for i in WORD_NS.split(',')]
    CHAR_NS = args.char_ngrams[1:-1]
    if len(CHAR_NS) == 0:
        CHAR_NS = []
    else:
        CHAR_NS = [int(i) for i in CHAR_NS.split(',')]
    print("Word-level n-grams used: " + str(WORD_NS) if not args.bpe else "-")
    print("Char-level n-grams used: " + str(CHAR_NS) if not args.bpe else "-")
    print("Adding uncased features: " + str(args.add_uncased))
    print("Using only uncased features: " + str(args.uncased_only))
    print("Using BPE tokenization for embeddings: " + str(args.bpe))

    featuremap = {}
    if args.single_utterance:
        if args.bpe:
            if 'flaubert' in args.embedding_model:
                tokenizer = FlaubertTokenizer.from_pretrained(
                    args.embedding_model,
                    do_lowercase='uncased' in args.embedding_model)
            else:
                tokenizer = BertTokenizer.from_pretrained(
                    args.embedding_model,
                    do_lowercase='uncased' in args.embedding_model)
            tokenizer.add_tokens(ESCAPE_TOKS, special_tokens=True)
            utterance2bpe_toks(tokenizer, args.single_utterance, '',
                               None, featuremap, verbose=True)
        else:
            utterance2ngrams(args.single_utterance, None, None, WORD_NS,
                             CHAR_NS, featuremap, verbose=True)
        sys.exit()

    if args.input_file:
        infile = args.input_file
    else:
        infile = 'data/bokmaal+phon_cleaned.tsv' \
                 if DIALECTS else 'data/tweets_cleaned.tsv'
    label_col = 0 if DIALECTS else 1
    data_col = 4 if DIALECTS else 2
    data = pd.read_csv(infile, encoding='utf8', delimiter='\t',
                       usecols=[label_col, data_col],
                       names=['labels', 'utterances'],
                       quoting=3  # "QUOTE_NONE"
                       )
    print(len(data))
    print(data['labels'].value_counts())

    outfile = args.model + '/features.tsv'
    Path(args.model).mkdir(parents=True, exist_ok=True)
    print("Writing features to", outfile)

    print("Extracting and saving the features.")
    if args.bpe:
        with open(outfile, 'w+', encoding='utf8') as f:
            f.write('utterance\tlabel\tBPE tokens\n')
        if 'flaubert' in args.embedding_model:
            tokenizer = FlaubertTokenizer.from_pretrained(
                args.embedding_model,
                do_lowercase='uncased' in args.embedding_model)
        else:
            tokenizer = BertTokenizer.from_pretrained(
                args.embedding_model,
                do_lowercase='uncased' in args.embedding_model)
        tokenizer.add_tokens(ESCAPE_TOKS, special_tokens=True)
        for utterance, label in zip(data['utterances'], data['labels']):
            _, featuremap = utterance2bpe_toks(tokenizer, utterance, label, outfile,
                               featuremap)
    else:
        with open(outfile, 'w+', encoding='utf8') as f:
            f.write('utterance\tlabel')
            for n in WORD_NS:
                f.write('\tword-' + str(n))
            for n in CHAR_NS:
                f.write('\tchar-' + str(n))
            f.write('\n')
        for utterance, label in zip(data['utterances'], data['labels']):
            _, featuremap = utterance2ngrams(utterance, label, outfile, WORD_NS, CHAR_NS,
                             featuremap)

    print("Saving the feature->context maps.")
    threshold = 10
    for label, feature2context in featuremap.items():
        filename = '{}/featuremap-{}.tsv'.format(args.model, label)
        print("Writing", filename)
        with open(filename, 'w+', encoding='utf8') as f:
            for feature, context in feature2context.items():
                f.write(feature)
                div = 1
                n_appearances = len(context)
                if args.percentage:
                    div = n_appearances
                n_common = 0
                for c, nr in Counter(context).most_common():
                    if nr < threshold:
                        break
                    n_common += nr
                    f.write('\t{}({:.1f})'.format(c, nr / div))
                if n_common > 0:
                    f.write('\t')
                f.write('below_threshold({:.1f})\n'.format(
                    (n_appearances - n_common) / n_appearances))
