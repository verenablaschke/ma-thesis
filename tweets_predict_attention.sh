#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets.sh

# get & clean tweets via the tweets_predict.sh script
python3 extract_features.py tweets models/tweets-attn --bpe --i data/tweets_cleaned_websites.tsv --embmod flaubert/flaubert_large_cased --perc
python3 feature_correlation.py models/tweets-attn
python3 prepare_folds.py models/tweets-attn 10

for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets-attn-$i
    screen -S tweets-attn-$i -X stuff "taskset -c 0-34 python3 predict_fold.py models/tweets-attn tweets $i --embed --log --mlm ffnn-attn rnn-attn --h 128 --ep 35 --b 64 --lr 0.01 0.01 --drop 0.4 --embmod word2vec --emblen 60\n"
done
