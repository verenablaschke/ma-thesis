#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets.sh


# for i in $(seq 0 9); do
#     echo "Setting up fold $i"
#     screen -dmS tweets$i
#     screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-embed tweets 0 1 2 3 4 5 6 7 8 9 --embed --log --mlm nn nn-attn --h 128 256 512 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4\n"
# done


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS tweets$i
    screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-embed tweets 0 1 2 3 4 5 6 7 8 9 --embed --log --mlm nn nn-attn --h 128 256 512 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4\n"
    for h in 128 256 512; do
        for ep in 5 15 25 35; do
            for b in 128; do
                for lr in 1 10; do
                    for d in 0 20 40; do
                        screen -S tweets$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-attn-h${h}-b${b}-d${d}-ep${ep}-em20-lr${lr}.txt\n"
                        screen -S tweets$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-h${h}-b${b}-d${d}-ep${ep}-em20-lr${lr}.txt\n"
                    done
                done
            done
        done
    done
done
