#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets.sh

# python3 get_tweets.py
python3 tweet_cleanup.py
python3 extract_features.py tweets models/tweets-z --lower
python3 feature_correlation.py models/tweets-z
python3 prepare_folds.py models/tweets-z 10
python3 predict_fold.py models/tweets-z tweets 0 --save --z 1000 --out '1000-0'


for i in $(seq 1 9); do
    echo "Setting up fold 1000-$i"
    screen -dmS tweets$i
    screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-z tweets $i --save --z 1000 --out '1000-$i'
"
done

# for i in $(seq 0 9); do
#     echo "Setting up fold 3000-$i"
#     screen -dmS tweets$i
#     screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-z tweets $i --save --z 3000 --out '3000-$i'
# "
# done

# for i in $(seq 0 9); do
#     echo "Setting up fold 5000-$i"
#     screen -dmS tweets$i
#     screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-z tweets $i --save --z 5000 --out '5000-$i'
# "
# done
