#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets_predict.sh

python3 get_tweets.py
python3 tweet_cleanup.py
python3 extract_features.py tweets models/tweet --bpe --i data/tweets_cleaned_websites.tsv --embmod flaubert/flaubert_large_cased
python3 feature_correlation.py models/tweets
python3 prepare_folds.py models/tweets 10


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets-embed-$i
    screen -S tweets-embed-$i -X stuff "python3 predict_fold.py models/tweets tweets $i --mlm svm --z 2000\n"
done
