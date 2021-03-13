# coding: utf-8

import pandas as pd
from pathlib import Path
import re
import sys
import argparse
from collections import Counter
from transformers import FlaubertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, help="'dialects' or 'tweets'")
parser.add_argument('model')
parser.add_argument('--word', dest='word_ngrams', default='[1,2]', type=str)
parser.add_argument('--char', dest='char_ngrams', default='[2,3,4,5]',
                    type=str)
parser.add_argument('--lower', dest='add_uncased', default=False,
                    action='store_true')
parser.add_argument('--embed', dest='use_embeddings', default=False,
                    action='store_true')
parser.add_argument('--embmod', dest='embedding_model',
                    default='flaubert/flaubert_base_cased', type=str)
args = parser.parse_args()

if args.type == 'dialects':
    DIALECTS = True
elif args.type == 'tweets':
    DIALECTS = False
else:
    print("Expected 'dialects' or 'tweets'")
    sys.exit()

if not args.model:
    print("You need to provide a path to the model to save or load it")
    sys.exit()


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
print("Word-level n-grams used: " + str(WORD_NS)
      if not args.use_embeddings else "-")
print("Char-level n-grams used: " + str(CHAR_NS)
      if not args.use_embeddings else "-")
print("Adding uncased features: " + str(args.add_uncased))
print("Using embeddings: " + str(args.use_embeddings))


featuremap = {}


def add_ngram(featuremap, label, ngram, context):
    try:
        featuremap[label][ngram].append(context)
    except KeyError:
        try:
            featuremap[label][ngram] = [context]
        except KeyError:
            featuremap[label] = {ngram: [context]}


def update_featuremap(featuremap, label, ngram, context):
    add_ngram(featuremap, label, ngram, context)
    add_ngram(featuremap, 'all', ngram, context)


# U0329, U030D are the combining lines for marking syllabic consonants
char_pattern = re.compile(r'(\w[\u0329\u030D]*|\.\w)',
                          re.UNICODE | re.IGNORECASE)


def utterance2ngrams(utterance, label, outfile, word_ns=WORD_NS,
                     char_ns=CHAR_NS, verbose=False):
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
    escape_toks = ['<URL>', '<USERNAME>', '<HASHTAG>', '<NUMBER>']
    sep = '<SEP>'
    for word_n in word_ns:
        cur_ngrams = []
        for i in range(len(words_wordlvl) + 1 - word_n):
            tokens = words_wordlvl[i:i + word_n]
            if not DIALECTS:
                tokens = [tok if tok in escape_toks else tok.lower()
                          for tok in tokens]
            ngram = sep.join(tokens)
            # Padding to distinguish these from char n-grams
            if args.add_uncased:
                uncased_ngram = 'UNCASED:<SOS>' + ngram.lower() + '<EOS>'
                cur_ngrams.append(uncased_ngram)
                update_featuremap(featuremap, label, uncased_ngram,
                                  '<SOS>' + ngram + '<EOS>')
            ngram = '<SOS>' + ngram + '<EOS>'
            cur_ngrams.append(ngram)
        ngrams.append(cur_ngrams)
    for char_n in char_ns:
        cur_ngrams = []
        for word, context in zip(words_charlvl, words_wordlvl):
            if word in escape_toks:
                continue
            chars = list(char_pattern.findall(word))
            word_len = len(chars)
            if word_len == 0:
                continue
            if char_n > 1:
                pfx = chars[:char_n - 1]
                ngram = sep + ''.join(pfx)
                cur_ngrams.append(ngram)
                update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    update_featuremap(featuremap, label,
                                      uncased_ngram, context)
            for i in range(len(chars) + 1 - char_n):
                ngram = ''.join(chars[i:i + char_n])
                cur_ngrams.append(ngram)
                update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    update_featuremap(featuremap, label,
                                      uncased_ngram, context)

            if char_n > 1:
                sfx = chars[word_len + 1 - char_n:]
                ngram = ''.join(sfx) + sep
                cur_ngrams.append(ngram)
                update_featuremap(featuremap, label, ngram, context)
                if args.add_uncased:
                    uncased_ngram = 'UNCASED:' + ngram.lower()
                    cur_ngrams.append(uncased_ngram)
                    update_featuremap(featuremap, label,
                                      uncased_ngram, context)
        ngrams.append(cur_ngrams)
    if verbose:
        print(utterance, ngrams)
    ngrams_flat = []
    with open(outfile, 'a', encoding='utf8') as f:
        f.write(utterance + '\t' + str(label))
        for lvl_ngrams in ngrams:
            ngrams_flat += lvl_ngrams
            f.write('\t')
            f.write(' '.join(lvl_ngrams))
        f.write('\n')
    return ngrams_flat


def utterance2bpe_toks(flaubert_tokenizer, utterance, label, outfile):
    # Based on the _tokenize method:
    # https://github.com/huggingface/transformers/blob/c89bdfbe720bc8f41c7dc6db5473a2cb0955f224/src/transformers/models/flaubert/tokenization_flaubert.py#L113
    # With the difference that I want to keep track of which token
    # a given BPE subtoken comes from.
    text = flaubert_tokenizer.preprocess_text(utterance)
    text = flaubert_tokenizer.moses_pipeline(text, lang="fr")
    text = flaubert_tokenizer.moses_tokenize(text, lang="fr")
    toks = []
    for token in text:
        if token:
            subtokens = [t for t in flaubert_tokenizer.bpe(token).split(" ")]
            for subtok in subtokens:
                update_featuremap(featuremap, label, subtok, token)
            toks.extend(subtokens)
    with open(outfile, 'a', encoding='utf8') as f:
        f.write("{}\t{}\t{}\n".format(utterance, label, ' '.join(toks)))
    return toks


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

# Extract and save the features.
if args.use_embeddings:
    with open(outfile, 'w+', encoding='utf8') as f:
        f.write('utterance\tlabel\tBPE tokens\n')
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(
        args.embedding_model, do_lowercase=False)
    for utterance, label in zip(data['utterances'], data['labels']):
        utterance2bpe_toks(flaubert_tokenizer, utterance, label, outfile)
else:
    with open(outfile, 'w+', encoding='utf8') as f:
        f.write('utterance\tlabel')
        for n in WORD_NS:
            f.write('\tword-' + str(n))
        for n in CHAR_NS:
            f.write('\tchar-' + str(n))
        f.write('\n')
    for utterance, label in zip(data['utterances'], data['labels']):
        utterance2ngrams(utterance, label, outfile)

# Save the feature->context map.
threshold = 10
for label, feature2context in featuremap.items():
    with open('{}/featuremap-{}.tsv'.format(args.model, label), 'w+',
              encoding='utf8') as f:
        for feature, context in feature2context.items():
            f.write(feature)
            filtered_context = ['{}({})'.format(cont, nr)
                                for cont, nr in Counter(context).most_common()
                                if nr >= threshold]
            if filtered_context:
                f.write('\t')
                f.write('\t'.join(filtered_context))
            f.write('\n')
