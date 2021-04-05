import html
import re
import unicodedata

chars = set()
with open('data/tweets.tsv', 'r', encoding='utf8') as f_in:
    with open('data/tweets_cleaned.tsv', 'w+', encoding='utf8') as f_out:
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
            # URLs: of the form abc.de; start with http(s):// or www or contain a /
            tweet = re.sub(
                '((?<=^)|(?<=\W))((https?://|www\d{0,3}\.)'
                '[a-zA-Z0-9.\-]+\.[a-z]{2,}|[a-zA-Z0-9.\-]+\.[a-z]{2,}/)'
                '([a-zA-Z0-9/\?%\+#~\.\-@\*!\(\)\[\]=:;,&\$/\']*)?',
                '<URL>', tweet)
            # Hashtags, numbers
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
