#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' tweets_analyze.sh

run_name="tweets"
run_name="tweets-may"

for label in 0 1; do
    echo "$label"
    screen -dmS ${run_name}-$label
    for type in all pos falsepos truepos; do
        echo "- $type"
        for comb in sqrt mean; do
            echo "-- $comb"
            screen -S ${run_name}-${label} -X stuff "python3 parse_results.py models/${run_name} ${label} ${type} 10 --comb ${comb}\n"
            screen -S ${run_name}-${label} -X stuff "python3 correlate_results.py models/${run_name}/importance_values_${comb}_${label}_${type}_unscaled_sorted.tsv models/${run_name}/features-correlated.tsv\n"
            screen -S ${run_name}-${label} -X stuff "python3 parse_results.py models/${run_name} ${label} ${type} 10 --comb ${comb} --scale\n"
            screen -S ${run_name}-${label} -X stuff "python3 correlate_results.py models/${run_name}/importance_values_${comb}_${label}_${type}_scaled_sorted.tsv models/${run_name}/features-correlated.tsv\n"
        done
    done
done

screen -S ${run_name}-0 -X stuff "python3 representativeness_distinctiveness.py models/${run_name}\n"
screen -S ${run_name}-0 -X stuff "python3 feature_context.py models/${run_name} tweets --comb mean --scores\n"
screen -S ${run_name}-0 -X stuff "python3 feature_context.py models/${run_name} tweets --t 200 --comb mean --r\n"
screen -S ${run_name}-0 -X stuff "python3 feature_context.py models/${run_name} tweets --t 21000 --comb mean --r\n"
screen -S ${run_name}-0 -X stuff "python3 feature_context.py models/${run_name} tweets --t 50 --comb mean --r\n"
screen -S ${run_name}-0 -X stuff "python3 feature_context.py models/${run_name} tweets --t 200 --comb sqrt --r\n"
screen -S ${run_name}-0 -X stuff "python3 plot_importance.py models/${run_name} tweets --comb mean --top 50 --topinput 200 --label\n"
screen -S ${run_name}-0 -X stuff "python3 plot_importance.py models/${run_name} tweets --comb mean --label\n"
