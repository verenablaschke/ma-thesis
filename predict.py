import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ngram_lime.lime.lime_text import LimeTextExplainer
import re
import sys
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('type', type=str,
                    help="'dialects' or 'tweets'")
parser.add_argument('model')
parser.add_argument('--start', dest='start_idx', default=0, type=int)
parser.add_argument('--stop', dest='stop_idx', default=-1, type=int)
parser.add_argument('--load', dest='load_model', type=bool, default=False)
args = parser.parse_args()


if args.type == 'dialects':
    DIALECTS = True
elif args.type == 'tweets':
    DIALECTS = False
else:
    print("Expected 'dialects' or 'tweets'")
    sys.exit()
LINEAR_SVC = DIALECTS

if args.load_model and not args.model:
    print("You need to provide a path to the model to load it")
    sys.exit()


np.random.seed(42)
SAVE_LOC = args.model


# U0329, U030D are the combining lines for marking syllabic consonants
char_pattern = re.compile(r'(\w[\u0329\u030D]*|\.\w)', re.UNICODE | re.IGNORECASE)


GET_NGRAMS = True

# def remove_ngrams(utterance, ngrams, remove_indices):
#     cleaned_utterances = []
#     for idx in remove_indices:
#         ngram = ngrams[idx]
#         # TODO works for the dialect data but likely not other input
#         placeholder = '?' * len(ngram)
#         cleaned_utterances.append(utterance.replace(ngram, placeholder))
#     combined_utterance = list(utterance)
#     for i in range(len(utterance)):
#         for utt in cleaned_utterances:
#             if utt[i] == '?':
#                 combined_utterance[i] = '?'
#                 break
#     return re.sub('\?+', '?', ''.join(combined_utterance))
    
# def utterance2ngrams(utterance, word_ns=[1, 2], char_ns=[1, 2, 3, 4, 5], verbose=False):
def utterance2ngrams(utterance, word_ns=[1, 2], char_ns=[2, 3, 4, 5], verbose=False):
    utterance = utterance.strip()
    if DIALECTS:
        words = utterance.split()
    else:
        # utterance = utterance.lower()
        utterance = utterance.replace(',', '')
        utterance = utterance.replace(';', '')
        utterance = utterance.replace('.', '')
        utterance = utterance.replace('’', "'")
        words = utterance.split()

    if not GET_NGRAMS:
        # print('WORDS', words)
        return words

    ngrams = []
    escape_toks = ['URL', 'USERNAME']
#     unk = 'UNK'  # none of the (uppercase!) letters appear in the data
    # unk = '?' # TODO works for the dialect data but likely not other input
    # words = utterance.split('#')
    sep = '<SEP>'
    for word_n in word_ns:
        for i in range(len(words) + 1 - word_n):
            tokens = words[i:i + word_n]
            if not DIALECTS:
                tokens = [tok if tok in escape_toks else tok.lower() for tok in tokens]
            ngram = sep.join(tokens)
            # if unk in ngram:
            #     continue
            ngrams.append('<SOS>' + ngram + '<EOS>')  # Padding to distinguish these from char n-grams
    for char_n in char_ns:
        for word in words:
            if word in escape_toks:
                continue
            chars = list(char_pattern.findall(word))
            word_len = len(chars)
            if word_len == 0:
                continue
                
            pfx = chars[:char_n - 1]
            # if unk in pfx:
            #     break
            ngrams.append(sep + ''.join(pfx))
            
            for i in range(len(chars) + 1 - char_n):
                ngram = chars[i:i + char_n]
                # if unk in ngram:
                #     continue
                ngrams.append(''.join(ngram))

            sfx = chars[word_len + 1 - char_n:]
            # if unk in sfx:
            #     break
            ngrams.append(''.join(sfx) + sep)
    if verbose:
        print(utterance, ngrams)
    # print('NGRAMS', ngrams)
    return ngrams


def parse_file(filename, test_size=0.2, max_features=5000, label_col=0, data_col=4,
               analyzer=utterance2ngrams):
    data = pd.read_csv(filename, encoding='utf8', delimiter='\t',
                       usecols=[label_col, data_col], names=['labels', 'utterances'])
    print(len(data))
    print(data['labels'].value_counts())
#     data['utterances'] = data['utterances'].map(preprocess)
#     data.dropna(inplace=True)
#     print(len(data))
#     print(data['labels'].value_counts())

    train_x_raw, test_x_raw, train_y, test_y = model_selection.train_test_split(
        data['utterances'], data['labels'], test_size=test_size,
        random_state=42)
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.transform(test_y)

#     vectorizer = TfidfVectorizer(max_features=max_features,
#                                  ngram_range=(1,2),
#                                  analyzer='word',
#                                  lowercase=False,  # Tjukk L -> uppercase L
#                                  token_pattern=r"\b[\w']+\b"  # ' marks syllabic consonants
#                                  )
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 analyzer=analyzer
                                 )
#     vectorizer = TfidfVectorizer(max_features=max_features,
#                                  ngram_range=(1,5),
#                                  analyzer='char',
#                                  lowercase=False,  # Tjukk L -> uppercase L
#                                  token_pattern=r"\b[\w']+\b"  # ' marks syllabic consonants
#                                  )
    train_x = vectorizer.fit_transform(train_x_raw)
    test_x = vectorizer.transform(test_x_raw)

    return train_x, test_x, list(train_x_raw), list(test_x_raw), train_y, test_y, label_encoder, vectorizer


def preprocess_and_vectorize(utterance, vectorizer):
    return vectorizer.transform([utterance])
    # return vectorizer.transform([preprocess(utterance)])


def train(train_x, train_y, linear_svc, class_weight=None):
    if linear_svc:
        model = svm.LinearSVC(C=1.0, class_weight=class_weight)
    else:
        # Binary cases
        model = svm.SVC(C=1.0, probability=True, class_weight=class_weight)
    model.fit(train_x, train_y)
    return model


def predict(model, test_x):
    return model.predict(test_x)


def predict_instance(model, utterance, label_encoder, vectorizer, linear_svc):
    # x = vectorizer.transform([preprocess(utterance)])
    x = vectorizer.transform([utterance])
    pred = model.predict(x)
    margins = model.decision_function(x)
    if linear_svc:
        exp = np.exp(margins)
        softmax = exp / np.sum(exp)
    #     print("{}: {}".format(utterance, softmax.round(3)))
        return pred[0], label_encoder.inverse_transform(pred)[0], margins, softmax
    return pred[0], label_encoder.inverse_transform(pred)[0], margins, model.predict_proba(x)


# def predict_proba(model, data, vectorizer, n_labels=4):
#     if isinstance(data, str):
#         data = [data]
#     probs = np.zeros((len(data), n_labels))
#     for i, utterance in enumerate(data):
#         x = preprocess_and_vectorize(utterance, vectorizer)
#         pred = model.predict(x)
#         margins = model.decision_function(x)
#         exp = np.exp(margins)
#         probs[i] = exp / np.sum(exp)  # softmax
#     return probs


def predict_proba2(model, data, vectorizer, linear_svc, n_labels=4):
    global GET_NGRAMS
    # print('data', data)
    probs = np.zeros((len(data), n_labels))
    GET_NGRAMS = False
    for i, utterance in enumerate(data):
        # utterance = [utterance.split()]
        # print(utterance)
        x = vectorizer.transform([utterance])
        if linear_svc:
            pred = model.predict(x)
            margins = model.decision_function(x)
            exp = np.exp(margins)
            probs[i] = exp / np.sum(exp)  # softmax
        else:
            probs[i] = model.predict_proba(x)
        # print('-' + str(i))
        # print(pred)
        # print(probs[i])
        # print(margins)
        # print(utterance)
        # # print(x)
    GET_NGRAMS = True
    return probs


def score(pred, test_y):
    print('Accuracy', accuracy_score(test_y, pred))
    print('F1 macro', f1_score(test_y, pred, average='macro'))
    print('Confusion matrix')
    print(confusion_matrix(test_y, pred))


def support_vectors(model, label_encoder, train_x, train_x_raw,
                    labels=[0, 1, 2, 3]):
    for label in labels:
        print(label_encoder.inverse_transform([label])[0])
        print('==========================\n')
        dec_fn = model.decision_function(train_x)[:, label]
        # support vectors and vectors 'in the middle of the street'
        support_vector_indices_pos = np.where(np.logical_and(dec_fn > 0, dec_fn <= 1))[0]
        print('positive side')
        for idx in support_vector_indices_pos:
            print('-', train_x_raw[idx])
        support_vector_indices_neg = np.where(np.logical_and(dec_fn <= 0, dec_fn >= -1))[0]
        print()
        print('negative side')
        for idx in support_vector_indices_pos:
            print('-', train_x_raw[idx])
        print('\n\n')


def instances_far_from_decision_boundary(model, label_encoder, train_x,
                                         train_x_raw, labels=[0, 1, 2, 3]):
    for label in labels:
        print(label_encoder.inverse_transform([label])[0])
        print('==========================\n')
        dec_fn = model.decision_function(train_x)[:, label]
        # vectors far away from the decision boundary
        support_vector_indices_pos = np.where(dec_fn > 3)[0]
        print('positive side')
        for idx in support_vector_indices_pos:
            print(dec_fn[idx].round(3), train_x_raw[idx])
    #     support_vector_indices_neg = np.where(dec_fn < -3)[0]
    #     print()
    #     print('negative side')
    #     for idx in support_vector_indices_pos:
    #         print('-', train_x_raw[idx])
        print('\n\n')




if args.load_model:
    print("Loading the model.")
    test_x_raw = []
    with open(SAVE_LOC + '/vectorizer.pickle', 'r', encoding='utf8') as f:
        for line in f:
            test_x_raw.append(line.strip())
    test_y = np.load(SAVE_LOC + '/test_y.npy')
    assert len(test_x_raw) == len(test_y)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(SAVE_LOC + '/label_encoder_classes.npy')
    with open(SAVE_LOC + '/vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(SAVE_LOC + '/classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    print("Done.")
else:
    if DIALECTS:
        FILE = 'data/phon_cleaned.tsv'
        # FILE = 'gdrive/My Drive/colab_projects/phon_cleaned.tsv'
    else:
        FILE = 'data/tweets_cleaned.tsv'
        # FILE = 'gdrive/My Drive/colab_projects/tweets_cleaned.tsv'    

    print("Preparing the data")
    label_col = 0 if DIALECTS else 1
    data_col = 4 if DIALECTS else 2
    train_x, test_x, train_x_raw, test_x_raw, train_y, test_y, label_encoder, vectorizer = parse_file(FILE, label_col=label_col, data_col=data_col)
    print("Training the model")
    class_weight = None if DIALECTS else {0: 1, 1: 2}
    classifier = train(train_x, train_y, linear_svc=LINEAR_SVC, class_weight=class_weight)
    print("Scoring the model")
    pred = predict(classifier, test_x)
    score(pred, test_y)

    print("Saving the model.")
    with open(SAVE_LOC + '/vectorizer.pickle', 'w', encoding='utf8') as f:
        for x in test_x_raw:
            f.write(x + '\n')
    np.save(SAVE_LOC + '/test_y.npy', test_y)
    np.save(SAVE_LOC + '/label_encoder_classes.npy', label_encoder.classes_)
    with open(SAVE_LOC + '/vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(SAVE_LOC + '/classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)



# # Roughly the baseline in the An Annotated Corpus... paper
# train_x_bs, test_x_bs, train_x_raw_bs, test_x_raw_bs, train_y_bs, test_y_bs, label_encoder_bs, vectorizer_bs = parse_file(FILE, label_col=1, data_col=2,
#     analyzer=lambda x: utterance2ngrams(x, word_ns=[1], char_ns=[], verbose=False))
# classifier_bs = train(train_x_bs, train_y_bs, class_weight={0: 1, 1: 2})
# pred_bs = predict(classifier_bs, test_x_bs)
# score(pred_bs, test_y_bs)
# support_vectors(classifier, label_encoder, train_x, train_x_raw)
# instances_far_from_decision_boundary(classifier, label_encoder, train_x, train_x_raw)


def save_to_file(filename, results_0, results_1, results_2, results_3):
    res = [results_0, results_1]
    if DIALECTS:
        res += [results_2, results_3]
    for i, r in enumerate(res):
        with open(filename.format(i), 'a', encoding='utf8') as f:
            for feature, score in r:
                f.write('{}\t{:.10f}\n'.format(feature, score))


labels = [0, 1, 2, 3] if DIALECTS else [0, 1]
explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform(labels),
                              split_expression='\s+', random_state=42, bow=True, 
                            #   mask_string='?',
                            #   remove_ngrams=remove_ngrams, 
                              utterance2ngrams=utterance2ngrams,
                              ngram_lvl=True,
                              recalculate_ngrams=False)
# lime_results = {0: dict(), 1: dict(), 2: dict(), 3: dict()}
results_0 = []
results_1 = []
results_2 = []
results_3 = []
filename = 'results/results_dialects_{}.txt' if DIALECTS else 'results/results_tweets_{}.txt'
for i, utterance in enumerate(test_x_raw[args.start_idx:args.stop_idx]):
    label = test_y[i]
    exp = explainer.explain_instance(utterance,
                                     lambda z: predict_proba2(classifier, z, vectorizer, LINEAR_SVC, n_labels=len(labels)),
                                     num_features=10,
                                     labels=labels,
                                     # labels=interesting_labels
                                    #  num_samples=5000
                                     )
            
    results_0 += exp.as_list(label=0)
    results_1 += exp.as_list(label=1)
    if DIALECTS:
        results_2 += exp.as_list(label=2)
        results_3 += exp.as_list(label=3)

    if i % 50 == 0:
        interesting_labels = [label]
        pred = predict_instance(classifier, utterance, label_encoder, vectorizer, LINEAR_SVC)
        print(i)
        print('"' + utterance + '""')
        print('ACTUAL', label_encoder.inverse_transform([label])[0])
        print('PREDICTED', pred[1])
        if pred[0] not in interesting_labels:
            interesting_labels.append(pred[0])
        # for l in interesting_labels:
        #     exp.show_in_notebook(text=utterance, labels=(l,))
        print('\n')
        save_to_file(filename, results_0, results_1, results_2, results_3)
        results_0 = []
        results_1 = []
        results_2 = []
        results_3 = []
    i += 1
    
save_to_file(filename, results_0, results_1, results_2, results_3)
