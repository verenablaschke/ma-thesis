import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('type', help="'dialects' or 'tweets'")
parser.add_argument('--t', dest='top', help="number of top-___ features",
                    default=100, type=int)
parser.add_argument('--c', dest='min_corr_info',
                    help="min NPMI correlation score for listing a correlated feature as ignored",
                    default=0.8, type=float)
parser.add_argument('--i', dest='min_corr_merge',
                    help="min NPMI correlation score for considering two features as identical",
                    default=1.0, type=float)
args = parser.parse_args()

threshold = args.top
labels = ['1', '0'] if args.type == 'tweets' else ['nordnorsk', 'vestnorsk',
                                                   'troendersk', 'oestnorsk']



feature2corr = {}
feature2identical = {}
with open('{}/features-correlated.tsv'.format(args.model),
              'r', encoding='utf8') as f:
    for line in f:
        feature1, feature2, corr = line.strip().split('\t')
        corr = float(corr)
        if corr < args.min_corr_info:
            # The file is sorted by correlation scores (descending order)
            break
        if corr < args.min_corr_merge:
            try:
                feature2corr[feature1][feature2] = corr
            except KeyError:
                feature2corr[feature1] = {feature2: corr}
            try:
                feature2corr[feature2][feature1] = corr
            except KeyError:
                feature2corr[feature2] = {feature1: corr}
        else:
            if feature1 == feature2:
                continue
            try:
                feature2identical[feature1].add(feature2)
            except KeyError:
                feature2identical[feature1] = {feature2}
            try:
                feature2identical[feature2].add(feature1)
            except KeyError:
                feature2identical[feature2] = {feature1}


for label in labels:
    feature2context = {}
    with open('{}/featuremap-{}.tsv'.format(args.model, label),
              'r', encoding='utf8') as f:
        for line in f:
            try:
                feature, context = line.strip().split('\t', maxsplit=1)
                context = context.replace('\t', ' ')
                feature2context[feature] = context
            except ValueError:
                # No frequent context
                pass
    feature2results = {}
    top_results = []
    with open('{}/importance_values_{}_all_sorted.tsv'.format(
              args.model, label), 'r', encoding='utf8') as f_in:
        header = next(f_in).strip()
        idx = 0
        for line in f_in:
            feature, mean, importance_sum, count = line.strip().split('\t')
            details = (idx, feature, float(mean), float(importance_sum), count,
                       feature2context.get(feature, ''),
                       feature2identical.get(feature, None),
                       feature2corr.get(feature, None))
            feature2results[feature] = details
            if idx < threshold:
                top_results.append(details)
            idx += 1


    with open('{}/importance_values_{}_all_sorted_{}context.tsv'.format(
            args.model, label, threshold), 'w+', encoding='utf8') as f_out:
        f_out.write('INDEX\t' + header + '\tCONTEXT\tIDENTICAL (IDX/FEATURE/MEAN/SUM/COUNT)\tCORRELATED (IDX/FEATURE/NPMI/MEAN/SUM/COUNT)\n')
        skip = set()
        for result in top_results:
            (idx, feature, mean, importance_sum, count, context, identical, correlated) = result
            if idx in skip:
                # Already listed
                continue

            f_out.write('{}\t{}\t{:.2f}\t{:.2f}\t{}\t{}\t'.format(
                idx, feature, mean, importance_sum, count, context))

            mirror_list = []
            if identical:
                for mirror in identical:
                    try:
                        (idx2, feature2, mean2, importance_sum2, count2, _, _, _) = feature2results[mirror]
                        if idx2 < threshold and idx2 > idx:
                            skip.add(idx)
                            print(feature, mirror)
                            print('Moved ' + str(idx2))
                        mirror_list.append('{}/{}/{:.2f}/{:.2f}/{}'.format(idx2, feature2, mean2, importance_sum2, count2))
                    except KeyError:
                        mirror_list.append('--/{}/--/--/--'.format(mirror))
            f_out.write(', '.join(mirror_list))
            f_out.write('\t')

            corr_list = []
            if correlated:
                for corr in correlated:
                    npmi = feature2corr[feature][corr]
                    try:
                        (idx2, feature2, mean2, importance_sum2, count2, _, _, _) = feature2results[corr]
                        corr_list.append('{}/{}/{:.2f}/{:.2f}/{:.2f}/{}'.format(idx2, feature2, npmi, mean2, importance_sum2, count2))
                    except KeyError:
                        corr_list.append('--/{}/{:.2f}/--/--/--'.format(corr, npmi))
            f_out.write(', '.join(corr_list))
            f_out.write('\n')


