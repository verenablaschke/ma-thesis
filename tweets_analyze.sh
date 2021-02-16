#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

for label in 0 1; do
    echo "$label"
    screen -dmS $label
    for type in all pos falsepos truepos; do
        echo "-- $type"
        screen -S $label -X stuff "python3 parse_results.py models/tweets $label $type 10
"
        screen -S $label -X stuff "python3 correlate_results.py models/tweets/importance_values_$label_$type_sorted.tsv models/tweets/features-correlated.tsv
"
    done
    for i in $(seq 0 9); do
        screen -S $label -X stuff "python3 parse_results.py models/tweets $label all $i --s
"
    done
done
