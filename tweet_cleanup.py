import html
import re
import unicodedata
import argparse
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from http.client import RemoteDisconnected, IncompleteRead

parser = argparse.ArgumentParser()
parser.add_argument('--i', dest='infile', default='data/tweets.tsv')
parser.add_argument('--o', dest='outfile', default='data/tweets_cleaned.tsv')
parser.add_argument('--url', dest='fetch_website_titles', default=False,
                    action='store_true')
parser.add_argument('--h', dest='leave_hashtags', default=False,
                    action='store_true')
args = parser.parse_args()


url_regex = '((?<=^)|(?<=\W))((https?://|www\d{0,3}\.)' \
            '[a-zA-Z0-9.\-]+\.[a-z]{2,}|[a-zA-Z0-9.\-]+\.[a-z]{2,}/)' \
            '([a-zA-Z0-9/\?%\+#~\.\-@\*!\(\)\[\]=:;,&\$/\']*)?'


def replace_url(tweet):
    match = re.search(url_regex, tweet)
    retrieved = False
    while match:
        url = match.group(0)
        title = ''
        try:
            soup = BeautifulSoup(urlopen(url), features="lxml")
            title = soup.title.string
        except Exception as e:
        # except (AttributeError, ValueError, HTTPError, URLError,
                # RemoteDisconnected, IncompleteRead):
            pass
        if title:
            retrieved = True
        else:
            title = ''
        tweet = tweet[:match.start(0)] + title + tweet[match.end(0):]
        match = re.search(url_regex, tweet)
    if retrieved:
        print(tweet)
    return tweet


chars = set()
with open(args.infile, 'r', encoding='utf8') as f_in:
    with open(args.outfile, 'w+', encoding='utf8') as f_out:
        prev_tweet = ''
        for line in f_in:
            line = line.strip()
            fields = line.split('\t', maxsplit=2)

            new_tweet = False
            if len(fields) == 3:
                try:
                    int(fields[0])
                    int(fields[1])
                    new_tweet = True
                except ValueError:
                    pass

            tweet = fields[2] if new_tweet else line
            # URLs: of the form abc.de; start with http(s):// or www or contain a /
            if args.fetch_website_titles:
                tweet = replace_url(tweet)
            else:
                tweet = re.sub(url_regex, '<URL>', tweet)
            # Replace non-breaking spaces etc.
            tweet = unicodedata.normalize("NFKC", tweet)
            # Replace, e.g., &gt with >
            tweet = html.unescape(tweet)
            # Zero-width characters
            tweet = re.sub('[\u200b\u200c\u200d\xad]', '', tweet)
            # Typographical variation
            tweet = re.sub('[«»“”„]|(´´)|(``)|( ́ ́)|( ̀ ̀)', '"',
                           tweet)
            tweet = re.sub("[\x92‘’´`]|('')|( ́)|( ̀)", "'", tweet)
            tweet = re.sub("''", '"', tweet)
            # Twitter usernames
            tweet = re.sub('((?<=^)|(?<=\W))@[a-zA-Z0-9_]+',
                           '<USERNAME>', tweet)
            # Hashtags, numbers
            if not args.leave_hashtags:
                tweet = re.sub('#\w+', '<HASHTAG>', tweet)
            tweet = re.sub('(?:(?<=\s)|(?<=^))[0-9]+(?:(?=\s)|(?=$))',
                           '<NUMBER>', tweet)

            if new_tweet:
                f_out.write(prev_tweet + '\n')
                prev_tweet = fields[0] + '\t' + fields[1] + '\t' + tweet
            else:
                prev_tweet += ' ' + tweet

            chars.update(tweet)

        if len(prev_tweet) > 0:
            f_out.write(prev_tweet + '\n')

chars = list(chars)
chars.sort()
with open('data/tweets_chars.tsv', 'w+', encoding='utf8') as f:
    f.write('{} characters\n{}\n'.format(len(chars), chars))
