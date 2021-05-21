#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets.sh

# # python3 get_tweets.py
# python3 tweet_cleanup.py
# python3 extract_features.py tweets models/tweets-embed --embed
# python3 feature_correlation.py models/tweets-embed
# python3 prepare_folds.py models/tweets-embed 10

# for i in $(seq 0 4); do
#     echo "Setting up fold $i"
#     screen -dmS tweets-embed-$i
#     screen -S tweets-embed-$i -X stuff "python3 predict_fold.py models/tweets-embed tweets $i --embed
# "
# done

# for i in $(seq 5 9); do
#     echo "Setting up fold $i"
#     screen -dmS tweets-embed-$i
#     screen -S tweets-embed-$i -X stuff "python3 predict_fold.py models/tweets-embed tweets $i --embed
# "
# done


# python3 get_tweets.py
# python3 tweet_cleanup.py
screen -dmS tweets-embed
screen -S tweets-embed -X stuff "python3 extract_features.py tweets models/tweets-may --bpe --i data/tweets_cleaned_websites.tsv --embmod flaubert/flaubert_large_cased\n"
screen -S tweets-embed -X stuff "python3 feature_correlation.py models/tweets-may\n"
screen -S tweets-embed -X stuff "python3 prepare_folds.py models/tweets-may 10\n"


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets-embed-$i
    screen -S tweets-embed-$i -X stuff "taskset -c 0-19 python3 predict_fold.py models/tweets-may tweets $i --mlm svm --z 2000\n"
done





for i in $(seq 1 9); do
    echo "Setting up fold $i"
    screen -dmS tweets-embed-$i
    screen -S tweets-embed-$i -X stuff "python3 predict_fold.py models/tweets-may tweets $i --embed --mlm nn\n"
done


python extract_features.py tweets models/tweets-website --bpe --perc