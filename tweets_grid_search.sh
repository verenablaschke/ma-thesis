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
    screen -S tweets$i -X stuff "python3 predict_fold.py models/tweets-embed tweets 0 1 2 3 4 5 6 7 8 9 --embed --log --nolime --mlm nn nn-attn --h 128 256 512 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4\n"
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


for i in $(seq 0 1); do
    echo "Setting up fold $i"
    screen -dmS tweets-large-40-$i
    screen -S tweets-large-40-$i -X stuff "python3 predict_fold.py models/tweets-embed tweets $i --embed --log --nolime --mlm nn nn-attn --h 128 256 512 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40\n"
    for h in 128 256 512; do
        for ep in 5 15 25 35; do
            for b in 128; do
                for lr in 1 10; do
                    for d in 0 20 40; do
                        screen -S tweets-large-40-$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-attn-h${h}-b${b}-d${d}-ep${ep}-T-20em768-lr${lr}.txt\n"
                        screen -S tweets-large-40-$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-h${h}-b${b}-d${d}-ep${ep}-em768-lr${lr}.txt\n"
                    done
                done
            done
        done
    done
done

for i in $(seq 1 1); do
    echo "Setting up fold $i"
    screen -dmS tweets-large-40-$i
    screen -S tweets-large-40-$i -X stuff "python3 predict_fold.py models/tweets-embed tweets $i --embed --log --nolime --mlm nn nn-attn --h 128 256 512 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40\n"
    for h in 128 256 512; do
        for ep in 5 15 25 35; do
            for b in 128; do
                for lr in 1 10; do
                    for d in 0 20 40; do
                        screen -S tweets-large-40-$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-attn-h${h}-b${b}-d${d}-ep${ep}-T-20em768-lr${lr}.txt\n"
                        screen -S tweets-large-40-$i -X stuff "python3 average_model_performance.py models/tweets-embed log-nn-h${h}-b${b}-d${d}-ep${ep}-em768-lr${lr}.txt\n"
                    done
                done
            done
        done
    done
done


taskset -c 0-19 python3 predict_fold.py models/tweets-embed tweets 8 --embed --log --nolime --mlm nn-attn ffnn-attn --h 128 256 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40 50 60

taskset -c 0-19 python3 predict_fold.py models/tweets-embed tweets 9 --embed --log --nolime --mlm nn-attn ffnn-attn --h 128 256 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40 50 60
taskset -c 0-19 python3 predict_fold.py models/tweets-embed tweets 5 --embed --log --nolime --mlm nn-attn ffnn-attn --h 128 256 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40 50 60


taskset -c 20-29 python3 predict_fold.py models/tweets-embed tweets 1 --embed --log --nolime --mlm nn-attn ffnn-attn --h 128 256 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod flaubert/flaubert_large_cased --emblen 40 50 60

taskset -c 20-29 python3 predict_fold.py models/tweets-embed tweets 7 --embed --log --nolime --mlm nn-attn nn --h 64 128 --ep 5 15 25 35 --b 128 --lr 0.001 0.01 --drop 0.0 0.2 0.4 0.6 --embmod flaubert/flaubert_large_cased --emblen 40 50 60



taskset -c 20-29 python3 predict_fold.py models/tweets-embed tweets 7 --embed --mlm nn-attn --h 128 --ep 25 --b 128 --lr 0.001 --drop 0.2 --embmod flaubert/flaubert_large_cased --emblen 40 --load_emb



renice -n 15 -p 87193

taskset -c 0-19 python3 predict_fold.py models/tweets-multi tweets 5 --embed --log --nolime --mlm rnn-attn --h 128 --ep 5 15 25 35 --b 16 --lr 0.001 0.01 --drop 0.0 0.2 0.4 --embmod bert-base-multilingual-cased --emblen 60 --load_emb
