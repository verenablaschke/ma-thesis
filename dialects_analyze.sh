#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

for label in nordnorsk vestnorsk oestnorsk troendersk; do
    echo "$label"
    screen -dmS $label
    for type in all pos falsepos truepos; do
        echo "- $type"
        for comb in sqrt mean; do
            echo "-- $comb"
            screen -S $label -X stuff "taskset -c 0-19 python3 parse_results.py models/dialects ${label} ${type} 10 --comb ${comb}\n"
            screen -S $label -X stuff "taskset -c 0-19 python3 correlate_results.py models/dialects/importance_values_${comb}_${label}_${type}_unscaled_sorted.tsv models/dialects/features-correlated.tsv\n"
            screen -S $label -X stuff "taskset -c 0-19 python3 parse_results.py models/dialects ${label} ${type} 10 --comb ${comb} --scale\n"
            screen -S $label -X stuff "taskset -c 0-19 python3 correlate_results.py models/dialects/importance_values_${comb}_${label}_${type}_scaled_sorted.tsv models/dialects/features-correlated.tsv\n"
        done
    done
done


python parse_results.py models/dialects nordnorsk all 10 --comb mean
python parse_results.py models/dialects vestnorsk all 10 --comb mean
python parse_results.py models/dialects oestnorsk all 10 --comb mean
python parse_results.py models/dialects troendersk all 10 --comb mean
