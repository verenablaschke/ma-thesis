python parse_results.py models\dialects nordnorsk all 10
python parse_results.py models\dialects oestnorsk all 10
python parse_results.py models\dialects vestnorsk all 10
python parse_results.py models\dialects troendersk all 10

python correlate_results.py models\dialects\importance_values_sqrt_nordnorsk_all_sorted.tsv models\dialects\features-correlated.tsv
python correlate_results.py models\dialects\importance_values_sqrt_vestnorsk_all_sorted.tsv models\dialects\features-correlated.tsv
python correlate_results.py models\dialects\importance_values_sqrt_oestnorsk_all_sorted.tsv models\dialects\features-correlated.tsv
python correlate_results.py models\dialects\importance_values_sqrt_troendersk_all_sorted.tsv models\dialects\features-correlated.tsv
