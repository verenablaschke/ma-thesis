import numpy as np
from transformers import FlaubertTokenizer
from extract_features import utterance2bpe_toks, get_escape_toks

ESCAPE_TOKS = get_escape_toks()

flaubert_tokenizer = FlaubertTokenizer.from_pretrained(
    'flaubert/flaubert_large_cased', do_lowercase=False)
flaubert_tokenizer.add_tokens(ESCAPE_TOKS, special_tokens=True)


char_len, word_len, bpe_tok_len = [], [], []

with open('data/tweets_cleaned.tsv', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tweet = line.split('\t')[2]
        tweet_wo_esc = tweet
        for esc in ESCAPE_TOKS:
            tweet_wo_esc = tweet_wo_esc.replace(esc, '*')
        char_len.append(len(tweet_wo_esc))
        if len(tweet_wo_esc) > 280:
            print(len(tweet_wo_esc))
            print(tweet_wo_esc)
        word_len.append(len(tweet.split(' ')))
        bpe_toks, _ = utterance2bpe_toks(flaubert_tokenizer, tweet, '', None, None)
        bpe_tok_len.append(len(bpe_toks))
        if len(flaubert_tokenizer.encode(tweet)) > 280:
            print(len(flaubert_tokenizer.encode(tweet)))
            print(tweet)


def stats(array, description, f):
    msg = '{}\tMean length\t{:.2f}'.format(description, np.mean(array))
    print(msg)
    f.write(msg + '\n')
    msg = '{}\tStd. deviation\t{:.2f}'.format(description, np.std(array))
    print(msg)
    f.write(msg + '\n')
    msg = '{}\tMin. value\t{}'.format(description, np.min(array))
    print(msg)
    f.write(msg + '\n')
    msg = '{}\tMax. value\t{}\n'.format(description, np.max(array))
    print(msg)
    f.write(msg)


with open('data/tweet_stats.tsv', 'w+', encoding='utf8') as f:
    n_tweets_msg = 'Number of tweets\t{:.2f}\n'.format(len(char_len))
    print(n_tweets_msg)
    f.write(n_tweets_msg)

    stats(char_len, 'Characters', f)
    stats(word_len, 'Tokens (whitespace-based)', f)
    stats(bpe_tok_len, 'Tokens (BPE-based)', f)
