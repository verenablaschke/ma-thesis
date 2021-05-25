#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets_analyze_attention.sh


python3 average_model_performance.py models/tweets-attn --log log-ffnn-attn-h128-b64-d40-ep35-T60-emword2vec-lr10.txt
python3 average_attention.py models/tweets-attn attention_scores-ffnn-attn-h128-b64-d40-ep35-T60-emword2vec-lr10.txt 60
