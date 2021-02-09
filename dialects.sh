#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

python3 extract_features.py dialects models/dialects --word "[1,2]" --char "[1,2,3,4,5]"
python3 feature_correlation.py models/dialects
python3 prepare_folds.py models/dialects 10

for i in $(seq 0 9); do
    echo "Setting up fold $i"
    screen -dmS dialects$i "python3 predict_fold.py models/dialects dialects $i
"
done

for label in nordnorsk vestnorsk oestnorsk troendersk; do
	echo "$label"
	screen -dmS $label
    for type in all pos falsepos truepos; do
    	echo "-- $type"
    	screen -S $label -X stuff "python3 parse_results.py models/dialects $label $type 10
"
    done
done
