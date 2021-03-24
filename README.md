
```
pip3 install -r requirements.txt

bash dialects_predict.sh
# Wait until this has finished running in all screens!
bash dialects_analyze.sh
# Wait until this has finished running in all screens!
python3 representativeness_specificity.py models/dialects
python3 feature_context.py models/dialects dialects --scores



python cluster_features.py models\dialects nordnorsk,vestnorsk,oestnorsk,troendersk 10 500
python feature_context.py models\dialects dialects
python cluster_features.py models\tweets 0,1 10 500
python representativeness_specificity.py models\tweets
python feature_context.py models\tweets tweets --scores
python plot_importance.py models\tweets\importance-spec-rep-sqrt.tsv


python feature_context.py models\tweets tweets --scores --comb mean
```