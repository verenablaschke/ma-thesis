#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets.sh

# python3 get_tweets.py
python3 tweet_cleanup.py
python3 extract_features.py tweets models/tweets --lower
python3 feature_correlation.py models/tweets
python3 prepare_folds.py models/tweets 10

for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets$i
    screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets tweets $i
"
done
