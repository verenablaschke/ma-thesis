#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

# screen -dmS dialects0
# screen -S dialects0 -X stuff "python3 extract_features.py dialects models/dialects-z --word "[1,2]" --char "[1,2,3,4,5]"
# "
# screen -S dialects0 -X stuff "python3 feature_correlation.py models/dialects-z
# "
# screen -S dialects0 -X stuff "python3 prepare_folds.py models/dialects-z 10
# "
# screen -S dialects0 -X stuff "python3 predict_fold.py models/dialects-z dialects 0 --save --z 1000 --out '1000-0' --limefeat 20 --v
# "


for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS dialects$i
    screen -S dialects$i -X stuff "python3 predict_fold.py models/dialects-z dialects 0 --load --z 200 --out '200-$i' --limefeat 100 --v\n"
    screen -S dialects$i -X stuff "python3 predict_fold.py models/dialects-z dialects 0 --load --z 300 --out '300-$i' --limefeat 100 --v\n"
    screen -S dialects$i -X stuff "python3 predict_fold.py models/dialects-z dialects 0 --load --z 500 --out '500-$i' --limefeat 100 --v\n"
done
