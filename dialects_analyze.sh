#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects_analyze.sh

for label in nordnorsk vestnorsk oestnorsk troendersk; do
    echo "$label"
    screen -dmS $label
    for type in all pos falsepos truepos; do
        echo "- $type"
        for comb in sqrt mean; do
            echo "-- $comb"
            screen -S $label -X stuff "python3 parse_results.py models/dialects ${label} ${type} 10 --comb ${comb}\n"
            screen -S $label -X stuff "python3 correlate_results.py models/dialects/importance_values_${comb}_${label}_${type}_unscaled_sorted.tsv models/dialects/features-correlated.tsv\n"
            screen -S $label -X stuff "python3 parse_results.py models/dialects ${label} ${type} 10 --comb ${comb} --scale\n"
            screen -S $label -X stuff "python3 correlate_results.py models/dialects/importance_values_${comb}_${label}_${type}_scaled_sorted.tsv models/dialects/features-correlated.tsv\n"
        done
    done
done

screen -S nordnorsk -X stuff "python3 feature_context.py models/dialects dialects --t 200 --comb mean --r\n"
screen -S nordnorsk -X stuff "python3 plot_importance.py models/dialects dialects --comb mean --top 50 --topinput 200 --label\n"
screen -S nordnorsk -X stuff "python3 plot_importance.py models/dialects dialects --comb mean\n"
