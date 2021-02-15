import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('type', help="'dialects' or 'tweets'")
parser.add_argument('--t', dest='top', help="number of top-___ features", type=int,
					default=100)
args = parser.parse_args()

threshold = args.top
labels = ['1', '0'] if args.type == 'tweets' else ['nordnorsk', 'vestnorsk',
											       'troendersk', 'oestnorsk']

for label in labels:
	feature2context = {}
	with open('{}/featuremap-{}.tsv'.format(args.model, label),
			  'r', encoding='utf8') as f:
		for line in f:
			try:
				feature, context = line.strip().split('\t', maxsplit=1)
				feature2context[feature] = context
			except ValueError:
				# No frequent context
				pass
	with open('{}/importance_values_{}_all_sorted.tsv'.format(
			args.model, label), 'r', encoding='utf8') as f_in:
		header = next(f_in).strip()
		with open('{}/importance_values_{}_all_sorted_{}context.tsv'.format(
				args.model, label, threshold), 'w+', encoding='utf8') as f_out:
			f_out.write(header + '\tCONTEXT\n')
			counter = 0
			for line in f_in:
				feature, mean, importance_sum, count = line.strip().split('\t')
				f_out.write('{}\t{}\t{}\t{}\t{}\n'.format(feature, mean,
						importance_sum, count,
						feature2context.get(feature, '')))
				counter += 1
				if counter == threshold:
					break
