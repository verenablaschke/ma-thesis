import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('type', help="'dialects' or 'tweets'")
parser.add_argument('--t', dest='top', help="number of top-___ features",
                    default=100, type=int)
parser.add_argument('--c', dest='min_corr_info',
                    help="min NPMI correlation score for listing a correlated"
                         " feature as ignored",
                    default=0.8, type=float)
parser.add_argument('--i', dest='min_corr_merge',
                    help="min NPMI correlation score for considering two "
                         "features as identical",
                    default=1.0, type=float)
parser.add_argument('--comb', dest='combination_method',
                    help='options: sqrt (square root of sums), mean',
                    default='sqrt', type=str)
args = parser.parse_args()

threshold = args.top
labels = ['1', '0'] if args.type == 'tweets' else ['nordnorsk', 'vestnorsk',
                                                   'troendersk', 'oestnorsk']

label2count = {}
with open(args.model + '/features.tsv', encoding='utf8') as f:
    next(f)  # Header
    for line in f:
        if len(line) == 0:
            continue
        label = line.split('\t')[1]
        try:
            label2count[label] += 1
        except KeyError:
            label2count[label] = 1
total_count = sum(val for val in label2count.values())
log_file = '{}/log_importance_values_{}_all_sorted_{}context.tsv' \
           .format(args.model, args.combination_method, threshold)
with open(log_file, 'w+', encoding='utf8') as f_log:
    f_log.write('LABEL\tPROPORTION\t'
                'IMPORTANCE_MEAN\tIMPORTANCE_VAR\t'
                'IMPORTANCE_MIN\tIMPORTANCE_MAX\t'
                'N_UTTERANCES_MEAN\tN_UTTERANCES_VAR\t'
                'N_UTTERANCES_MIN\tN_UTTERANCES_MAX\t'
                'REPRESENTATIVITY_MEAN\tREPRESENTATIVITY_VAR\t'
                'REPRESENTATIVITY_MIN\tREPRESENTATIVITY_MAX\t'
                'CORRCOEF_IMPORTANCE_REP\tCOVARIANCE_IMPORTANCE_REP\t'
                'SPECIFICITY_MEAN\tSPECIFICITY_VAR\t'
                'SPECIFICITY_MIN\tSPECIFICITY_MAX\t'
                'CORRCOEF_IMPORTANCE_SPEC\tCOVARIANCE_IMPORTANCE_SPEC\n')

with open(args.model + '/features.tsv', encoding='utf8') as f:
    next(f)  # Header
    for line in f:
        if len(line) == 0:
            continue
        label = line.split('\t')[1]
        try:
            label2count[label] += 1
        except KeyError:
            label2count[label] = 1
total_count = sum(val for val in label2count.values())

print("Reading the feature correlations.")
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
        if feature1 == feature2:
            continue
        if corr < args.min_corr_merge:
            try:
                feature2corr[feature1][feature2] = corr
            except KeyError:
                feature2corr[feature1] = {feature2: corr}
            try:
                feature2corr[feature2][feature1] = corr
            except KeyError:
                feature2corr[feature2] = {feature1: corr}
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
    print("LABEL", label)
    print("Getting the feature context.")
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
    with open('{}/importance_values_{}_{}_all_sorted.tsv'
              .format(args.model, args.combination_method, label),
              'r', encoding='utf8') as f_in:
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

    print("Reading the representativeness/specificity features")
    distribution = {}
    with open(args.model + '/feature-distribution.tsv', 'r',
              encoding='utf8') as f:
        cols = next(f).strip().split('\t')
        count_col = cols.index(label)
        rep_col = cols.index(label + '-REP')
        spec_col = cols.index(label + '-SPEC')
        next(f)  # Summary of the entire dataset.
        for line in f:
            cells = line.strip().split('\t')
            distribution[cells[0]] = (cells[count_col], cells[rep_col],
                                      cells[spec_col])

    imp_scores, n_utt_scores, rep_scores, spec_scores = [], [], [], []

    with open('{}/importance_values_{}_{}_all_sorted_{}context.tsv'
              .format(args.model, args.combination_method, label, threshold),
              'w+', encoding='utf8') as f_out:
        f_out.write('INDEX\t' + header + '\tCONTEXT\tN_UTTERANCES\t'
                    'REPRESENTATIVENESS\tSPECIFICITY\t'
                    'N_IDENTICAL_TOP\tIDENTICAL (IDX/FEATURE/MEAN/SUM/COUNT)\t'
                    'CORRELATED (IDX/FEATURE/NPMI/MEAN/SUM/COUNT)\n')
        skip = set()
        for result in top_results:
            (idx, feature, mean, importance_sum, count, context,
             identical, correlated) = result
            if idx in skip:
                # Already listed
                continue

            n_occ, rep, spec = distribution[feature]

            f_out.write('{}\t{}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t'.format(
                idx, feature, mean, importance_sum, count, context,
                n_occ, rep, spec))
            imp_scores.append(float(mean))
            n_utt_scores.append(int(float(n_occ)))
            rep_scores.append(float(rep))
            spec_scores.append(float(spec))

            mirror_list = []
            n_identical_top = 0
            if identical:
                for mirror in identical:
                    try:
                        (idx2, feature2, mean2, importance_sum2,
                         count2, _, _, _) = feature2results[mirror]
                        if idx2 < threshold and idx2 > idx:
                            n_identical_top += 1
                            skip.add(idx2)
                            print(feature, mirror)
                            print('Moved ' + str(idx2))
                        mirror_list.append('{}/{}/{:.2f}/{:.2f}/{}'.format(
                            idx2, feature2, mean2, importance_sum2, count2))
                    except KeyError:
                        mirror_list.append('--/{}/--/--/--'.format(mirror))
            f_out.write('{}\t{}\t'.format(n_identical_top,
                                          ', '.join(mirror_list)))

            corr_list = []
            if correlated:
                for corr in correlated:
                    npmi = feature2corr[feature][corr]
                    try:
                        (idx2, feature2, mean2, importance_sum2,
                         count2, _, _, _) = feature2results[corr]
                        corr_list.append(
                            (npmi, '{}/{}/{:.2f}/{:.2f}/{:.2f}/{}'.format(
                                idx2, feature2, npmi, mean2,
                                importance_sum2, count2)))
                    except KeyError:
                        corr_list.append(
                            (npmi, '--/{}/{:.2f}/--/--/--'.format(corr, npmi)))
                corr_list = [entry for (_, entry) in sorted(
                    corr_list, key=lambda x: x[0], reverse=True)]
            f_out.write(', '.join(corr_list))
            f_out.write('\n')

    with open(log_file, 'a', encoding='utf8') as f_log:
        f_log.write('{}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                    '\t{:.1f}\t{:.1f}\t{}\t{}'
                    '\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}'
                    '\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\n'
                    .format(label, label2count[label] / total_count,
                            np.mean(imp_scores), np.var(imp_scores),
                            np.min(imp_scores), np.max(imp_scores),
                            np.mean(n_utt_scores),
                            np.var(n_utt_scores),
                            np.min(n_utt_scores), np.max(n_utt_scores),
                            np.mean(rep_scores), np.var(rep_scores),
                            np.min(rep_scores), np.max(rep_scores),
                            np.corrcoef(imp_scores, rep_scores)[0, 1],
                            np.cov(imp_scores, rep_scores)[0, 1],
                            np.mean(spec_scores), np.var(spec_scores),
                            np.min(spec_scores), np.max(spec_scores),
                            np.corrcoef(imp_scores, spec_scores)[0, 1],
                            np.cov(imp_scores, spec_scores)[0, 1]))
