import keys

from datetime import datetime
import tweepy


auth = tweepy.AppAuthHandler(keys.consumer_key, keys.consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

counter = 0
n_skipped = 0
n_sexist = 0

with open("data/corpus_SexistContent.csv", "r", encoding="utf8") as in_file:
    with open("data/tweets.tsv", "w", encoding="utf8") as out_file:
        with open("data/skipped.tsv", "w", encoding="utf8") as skipped_file:
            for line in in_file:
                line = line.strip()
                cells = line.split('\t')  # ID, label
                msg = ""
                try:
                    # tweet_mode='extended' is crucial to get >140 chars
                    msg = api.get_status(cells[0],
                                         tweet_mode='extended').full_text
                    out_file.write(line + '\t' + msg + '\n')
                    if cells[1] == "1":
                        n_sexist += 1
                except Exception as e:
                    n_skipped += 1
                    msg = repr(e)
                    skipped_file.write(line + '\t' + msg + '\n')
                counter += 1
                if counter % 500 == 0:
                    print(datetime.now().strftime('%H:%M:%S'),
                          "(" + cells[1] + ")", msg)
                    print("Total: {}\tSkipped: {}\tSexist: {}"
                          .format(counter, n_skipped, n_sexist))