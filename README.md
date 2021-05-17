
```
pip3 install -r requirements.txt

bash dialects_predict.sh
# Wait until this has finished running in all screens!
bash dialects_analyze.sh
# Wait until this has finished running in all screens!
python3 representativeness_specificity.py models/dialects
python3 feature_context.py models/dialects dialects --scores
python3 feature_context.py models/dialects dialects --scores --comb mean



python cluster_features.py models\dialects nordnorsk,vestnorsk,oestnorsk,troendersk 10 500
python feature_context.py models\dialects dialects
python cluster_features.py models\tweets 0,1 10 500
python representativeness_specificity.py models\tweets
python feature_context.py models\tweets tweets --scores

python feature_context.py models\tweets tweets --scores --comb mean
python feature_context.py models\tweets tweets --scores --comb mean --scale
python feature_context.py models\tweets tweets --scores --comb sqrt
python feature_context.py models\tweets tweets --scores --comb sqrt --scale

python feature_context.py models\tweets tweets --scores --comb mean --m falsepos


python feature_context.py models\tweets tweets --scores --comb mean


python plot_importance.py models\tweets --comb mean
python plot_importance.py models\tweets --comb mean --scale
python plot_importance.py models\tweets --comb sqrt
python plot_importance.py models\tweets --comb sqrt --scale



python feature_context.py models\dialects dialects --scores --comb mean
python feature_context.py models\dialects dialects --scores --comb sqrt


python representativeness_specificity.py models\tweets-bpe
python feature_context.py models\tweets-bpe tweets --scores --comb sqrt

python plot_importance.py models\dialects --comb mean
python plot_importance.py models\dialects --comb sqrt

python compare_averages.py models\tweets\importance_values_sqrt_{}_all_unscaled_sorted.tsv models\tweets\importance_values_sqrt_{}_all_scaled_sorted.tsv


python compare_averages.py models\dialects\importance_values_sqrt_{}_all_unscaled_sorted.tsv models\dialects\importance_values_sqrt_{}_all_scaled_sorted.tsv
python compare_averages.py models\dialects\importance_values_mean_{}_all_unscaled_sorted.tsv models\dialects\importance_values_mean_{}_all_scaled_sorted.tsv
python compare_averages.py models\dialects\importance_values_mean_{}_all_unscaled_sorted.tsv models\dialects\importance_values_sqrt_{}_all_unscaled_sorted.tsv


python compare_lime_sample_sizes.py models/dialects-z/fold-0 dialects
python compare_lime_sample_sizes.py models/tweets-z/fold-0 tweets


python check_model_r2.py models/tweets/ tweets



python compare_averages.py models\tweets\importance_values_sqrt_{}_all_unscaled_sorted.tsv models\tweets\importance_values_sqrt_{}_falsepos_unscaled_sorted.tsv
```