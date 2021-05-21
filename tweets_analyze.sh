#!/bin/sh

# In the case of line ending problems (Windows vs. UNIX encodings), run:
# sed -i 's/\r$//' dialects.sh

run_name="tweets"
# run_name="tweets-cased"
# run_name="tweets-bpe"

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

screen -S ${run_name}-${label} -X stuff "python3 representativeness_specificity.py models/${run_name}\n"
screen -S ${run_name}-${label} -X stuff "python3 feature_context.py models/${run_name} tweets --scores --comb sqrt\n"
screen -S ${run_name}-${label} -X stuff "python3 feature_context.py models/${run_name} tweets --scores --comb mean\n"


run_name="tweets-website"
screen -S ${run_name}-${label} -X stuff "taskset -c 0-19 python3 representativeness_specificity.py models/${run_name}\n"
screen -S ${run_name}-${label} -X stuff "taskset -c 0-19 python3 feature_context.py models/${run_name} tweets --scores --comb sqrt\n"
screen -S ${run_name}-${label} -X stuff "taskset -c 0-19 python3 feature_context.py models/${run_name} tweets --scores --comb mean\n"

run_name="tweets-website"
taskset -c 0-19 python3 representativeness_specificity.py models/${run_name}
taskset -c 0-19 python3 feature_context.py models/${run_name} tweets --scores --comb sqrt
taskset -c 0-19 python3 feature_context.py models/${run_name} tweets --scores --comb mean


python3 parse_results.py models/tweets-website 0 all 10 --comb mean
python3 parse_results.py models/tweets-website 1 all 10 --comb mean
python3 parse_results.py models/tweets-website 0 all 10 --comb sqrt
python3 parse_results.py models/tweets-website 1 all 10 --comb sqrt
