python parse_results.py models\dialects10 oestnorsk pos 10
python parse_results.py models\dialects10 vestnorsk pos 10
python parse_results.py models\dialects10 troendersk pos 10

python parse_results.py models\dialects10 nordnorsk falsepos 10
python parse_results.py models\dialects10 oestnorsk falsepos 10
python parse_results.py models\dialects10 vestnorsk falsepos 10
python parse_results.py models\dialects10 troendersk falsepos 10

python parse_results.py models\dialects10 nordnorsk truepos 10
python parse_results.py models\dialects10 oestnorsk truepos 10
python parse_results.py models\dialects10 vestnorsk truepos 10
python parse_results.py models\dialects10 troendersk truepos 10


python correlate_results.py models\dialects10\importance_values_nordnorsk_pos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_nordnorsk_truepos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_nordnorsk_falsepos_sorted.tsv models\dialects10\features-correlated.tsv

python correlate_results.py models\dialects10\importance_values_vestnorsk_pos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_vestnorsk_truepos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_vestnorsk_falsepos_sorted.tsv models\dialects10\features-correlated.tsv

python correlate_results.py models\dialects10\importance_values_oestnorsk_pos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_oestnorsk_truepos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_oestnorsk_falsepos_sorted.tsv models\dialects10\features-correlated.tsv

python correlate_results.py models\dialects10\importance_values_troendersk_pos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_troendersk_truepos_sorted.tsv models\dialects10\features-correlated.tsv
python correlate_results.py models\dialects10\importance_values_troendersk_falsepos_sorted.tsv models\dialects10\features-correlated.tsv
