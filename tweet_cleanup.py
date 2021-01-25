import re

with open ('data/corpus_SexistContent_tweets.csv', 'r', encoding='utf8') as f_in:
    with open ('data/tweets_cleaned.tsv', 'w', encoding='utf8') as f_out:
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
            # Twitter usernames
            tweet = re.sub('((?<=^)|(?<=\W))@[a-zA-Z0-9_]+', '<USERNAME>', tweet)
            # URLs: of the form abc.de; start with http(s):// or www or contain a /
            tweet = re.sub('((?<=^)|(?<=\W))((https?://|www\d{0,3}\.)[a-zA-Z0-9.\-]+\.[a-z]{2,}|[a-zA-Z0-9.\-]+\.[a-z]{2,}/)([a-zA-Z0-9/\?%\+#~\.\-@\*!\(\)\[\]=:;,&\$/\']*)?', '<URL>', tweet)

            # print('NEW' if new_tweet else 'CONT', tweet)
            
            if new_tweet:
                f_out.write(prev_tweet + '\n')
                prev_tweet = fields[0] + '\t' + fields[1] + '\t' + tweet
            else:
                prev_tweet += ' ' + tweet


        f_out.write(prev_tweet + '\n')
