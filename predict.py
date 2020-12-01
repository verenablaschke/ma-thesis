import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
from ngram_lime.lime.lime_text import LimeTextExplainer
import re

np.random.seed(42)

DIALECTS = True
if DIALECTS:
    # FILE = 'data/phon_cleaned.tsv'
    FILE = 'gdrive/My Drive/colab_projects/phon_cleaned.tsv'
else:
    FILE = 'gdrive/My Drive/colab_projects/tweets_cleaned.tsv'    

# U0329, U030D are the combining lines for marking syllabic consonants
char_pattern = re.compile(r'(\w[\u0329\u030D]*|\.\w)', re.UNICODE | re.IGNORECASE)

LINEAR_SVC = DIALECTS


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
    words = utterance.split()
    if not GET_NGRAMS:
        # print('WORDS', words)
        return words

    ngrams = []
#     unk = 'UNK'  # none of the (uppercase!) letters appear in the data
    # unk = '?' # TODO works for the dialect data but likely not other input
    # words = utterance.split('#')
    sep = '<SEP>'
    for word_n in word_ns:
        for i in range(len(words) + 1 - word_n):
            ngram = sep.join(words[i:i + word_n])
            # if unk in ngram:
            #     continue
            ngrams.append('<SOS>' + ngram + '<EOS>')  # Padding to distinguish these from char n-grams
    for char_n in char_ns:
        for word in words:
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


def parse_file(filename, test_size=0.2, max_features=5000, label_col=0, data_col=4):
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
                                 analyzer=utterance2ngrams
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


def train(train_x, train_y, linear_svc=LINEAR_SVC):
    if linear_svc:
        model = svm.LinearSVC(C=1.0)
    else:
        # Binary cases
        model = svm.SVC(C=1.0, probability=True)
    model.fit(train_x, train_y)
    return model


def predict(model, test_x):
    return model.predict(test_x)


def predict_instance(model, utterance, label_encoder, vectorizer, linear_svc=LINEAR_SVC):
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


def predict_proba2(model, data, vectorizer, n_labels=4, linear_svc=LINEAR_SVC):
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
    print('Accuracy', accuracy_score(pred, test_y))
    print('F1 macro', f1_score(pred, test_y, average='macro'))


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

train_x, test_x, train_x_raw, test_x_raw, train_y, test_y, label_encoder, vectorizer = parse_file(FILE)
model = train(train_x, train_y)
pred = predict(model, test_x)
score(pred, test_y)
# support_vectors(model, label_encoder, train_x, train_x_raw)
# instances_far_from_decision_boundary(model, label_encoder, train_x, train_x_raw)



explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform([0, 1, 2, 3]),
                              split_expression='\s+', random_state=42, bow=True, 
                            #   mask_string='?',
                            #   remove_ngrams=remove_ngrams, 
                              utterance2ngrams=utterance2ngrams,
                              ngram_lvl=True,
                              recalculate_ngrams=False)
labels = [0, 1, 2, 3]
# lime_results = {0: dict(), 1: dict(), 2: dict(), 3: dict()}
results_0 = []
results_1 = []
results_2 = []
results_3 = []
for i, utterance in enumerate(test_x_raw):
    label = test_y[i]
    exp = explainer.explain_instance(utterance,
                                     lambda z: predict_proba2(model, z, vectorizer),
                                     num_features=10,
                                     labels=labels,
                                     # labels=interesting_labels
                                    #  num_samples=5000
                                     )
            
    results_0 += exp.as_list(label=0)
    results_1 += exp.as_list(label=1)
    results_2 += exp.as_list(label=2)
    results_3 += exp.as_list(label=3)

    if i % 50 == 0:
        interesting_labels = [label]
        pred = predict_instance(model, utterance, label_encoder, vectorizer)
        print(i)
        print('"' + utterance + '""')
        print('ACTUAL', label_encoder.inverse_transform([label])[0])
        print('PREDICTED', pred[1])
        if pred[0] not in interesting_labels:
            interesting_labels.append(pred[0])
        # for l in interesting_labels:
        #     exp.show_in_notebook(text=utterance, labels=(l,))
        print('\n')
    i += 1
    

filename = 'results_dialects_{}.txt' if DIALECTS else 'results_tweets_{}.txt'
for i, res in enumerate([results_0, results_1, results_2, results_3]):
    with open(filename.format(i), 'w', encoding='utf8') as f:
        for feature, score in res:
            f.write('{}\t{:.10f}\n'.format(feature, score))
