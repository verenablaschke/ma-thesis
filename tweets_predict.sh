#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets_predict.sh

python3 get_tweets.py
python3 tweet_cleanup.py
# python3 extract_features.py tweets models/tweets --bpe --i data/tweets_cleaned_websites.tsv --embmod flaubert/flaubert_large_cased --perc
# python3 feature_correlation.py models/tweets
# python3 prepare_folds.py models/tweets 10


# for i in $(seq 0 9); do
#     echo "Setting up fold $i"
#     screen -dmS tweets-$i
#     screen -S tweets-$i -X stuff "python3 predict_fold.py models/tweets tweets $i --mlm svm --z 2000\n"
# done


python3 extract_features.py tweets models/tweets-may-attn --bpe --i data/tweets_cleaned_websites.tsv --embmod flaubert/flaubert_large_cased --perc
python3 feature_correlation.py models/tweets-may-attn
python3 prepare_folds.py models/tweets-may-attn 10


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets-$i
    screen -S tweets-$i -X stuff "taskset -c 0-28 python3 predict_fold.py models/tweets-may-attn tweets $i --embed --log --mlm rnn-attn --h 128 --ep 35 --b 64 --lr 0.01 --drop 0.4 --embmod flaubert/flaubert_large_cased --emblen 60\n"
done


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    # screen -dmS tweets-$i
    screen -S tweets-$i -X stuff "taskset -c 0-28 python3 predict_fold.py models/tweets-may-attn tweets $i --embed --log --mlm ffnn-attn --h 128 --ep 35 --b 64 --lr 0.01 --drop 0.4 --embmod flaubert/flaubert_large_cased --emblen 60\n"
done


