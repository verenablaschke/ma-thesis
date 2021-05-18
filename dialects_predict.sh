#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

taskset -c 0-19 python3 extract_features.py dialects models/dialects --word "[1,2]" --char "[1,2,3,4,5]"
python extract_features.py dialects models/dialects --word "[1,2]" --char "[1,2,3,4,5]" --perc
taskset -c 0-19 python3 feature_correlation.py models/dialects
taskset -c 0-19 python3 prepare_folds.py models/dialects 10

for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS dialects$i
    screen -S dialects$i -X stuff "taskset -c 0-19 python3 predict_fold.py models/dialects-may dialects $i --z 1000 --limefeat 100
"
done
