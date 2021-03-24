#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

for label in 0 1; do
    echo "$label"
    screen -dmS $label
    for type in all pos falsepos truepos; do
        echo "- $type"
        for comb in sqrt mean; do
            echo "-- $comb"
            screen -S $label -X stuff "python3 parse_results.py models/tweets ${label} ${type} 10 --comb ${comb}\n"
            screen -S $label -X stuff "python3 correlate_results.py models/tweets/importance_values_${comb}_${label}_${type}_unscaled_sorted.tsv models/tweets/features-correlated.tsv\n"
            screen -S $label -X stuff "python3 parse_results.py models/tweets ${label} ${type} 10 --comb ${comb} --scale\n"
            screen -S $label -X stuff "python3 correlate_results.py models/tweets/importance_values_${comb}_${label}_${type}_scaled_sorted.tsv models/tweets/features-correlated.tsv\n"
        done
    done
done
