import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
# from lime.lime_text import LimeTextExplainer
from ngram_lime.lime.lime_text import LimeTextExplainer
import re


np.random.seed(42)


FILE = 'data/phon_parsed.tsv'


def preprocess(utterance):
    # TODO works for the dialect data but likely not other input
    return utterance.replace(' ', '#')


def remove_ngrams(utterance, ngrams, remove_indices):
    cleaned_utterances = []
    for idx in remove_indices:
        ngram = ngrams[idx]
        # TODO works for the dialect data but likely not other input
        placeholder = '?' * len(ngram)
        cleaned_utterances.append(utterance.replace(ngram, placeholder))
    combined_utterance = list(utterance)
    for i in range(len(utterance)):
        for utt in cleaned_utterances:
            if utt[i] == '?':
                combined_utterance[i] = '?'
                break
    return re.sub('\?+', '?', ''.join(combined_utterance))
    
# def utterance2ngrams(utterance, word_ns=[1, 2], char_ns=[1, 2, 3, 4, 5], verbose=False):
def utterance2ngrams(utterance, word_ns=[1, 2], char_ns=[2, 3, 4, 5], verbose=False):
    ngrams = []
#     unk = 'UNK'  # none of the (uppercase!) letters appear in the data
    unk = '?' # TODO works for the dialect data but likely not other input
    words = utterance.split('#')
    for word_n in word_ns:
        for i in range(len(words) + 1 - word_n):
            ngram = '#'.join(words[i:i + word_n])
            if unk in ngram:
                continue
            ngrams.append(ngram)
    for char_n in char_ns:
        for word in words:
            word_len = len(word)
            if word_len == 0:
                continue
                
            pfx = word[:char_n - 1]
            if '?' in pfx:
                break
            ngrams.append('#' + pfx)
            
            for i in range(len(word) + 1 - char_n):
                ngram = word[i:i + char_n]
                if unk in ngram:
                    continue
                ngrams.append(ngram)

            sfx = word[word_len + 1 - char_n:]
            if '?' in sfx:
                break
            ngrams.append(sfx + '#')
    if verbose:
        print(utterance, ngrams)
    return ngrams


def parse_file(filename, test_size=0.2, max_features=5000):
    data = pd.read_csv(filename, encoding='utf8', delimiter='\t',
                       usecols=[0, 4], names=['labels', 'utterances'])
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

    return train_x, test_x, train_x_raw, test_x_raw, train_y, test_y, label_encoder, vectorizer


def preprocess_and_vectorize(utterance, vectorizer):
    return vectorizer.transform([preprocess(utterance)])


def train(train_x, train_y):
    model = svm.LinearSVC(C=1.0)
    model.fit(train_x, train_y)
    return model


def predict(model, test_x):
    return model.predict(test_x)


def predict_instance(model, utterance, label_encoder, vectorizer):
    x = vectorizer.transform([preprocess(utterance)])
    pred = model.predict(x)
    margins = model.decision_function(x)
    exp = np.exp(margins)
    softmax = exp / np.sum(exp)
#     print("{}: {}".format(utterance, softmax.round(3)))
    return pred[0], label_encoder.inverse_transform(pred)[0], margins, softmax


def predict_proba(model, data, vectorizer):
    if isinstance(data, str):
        data = [data]
    probs = np.zeros((len(data), 4))
    for i, utterance in enumerate(data):
        x = preprocess_and_vectorize(utterance, vectorizer)
        pred = model.predict(x)
        margins = model.decision_function(x)
        exp = np.exp(margins)
        probs[i] = exp / np.sum(exp)  # softmax
    return probs


def predict_proba2(model, data, vectorizer):
    print('data', data)
    probs = np.zeros((len(data), 4))
    for i, utterance in enumerate(data):
        x = vectorizer.transform(utterance)
        pred = model.predict(x)
        margins = model.decision_function(x)
        exp = np.exp(margins)
        probs[i] = exp / np.sum(exp)  # softmax
    return probs


def score(pred, test_y):
    print('Accuracy', accuracy_score(pred, test_y))
    print('F1 macro', f1_score(pred, test_y, average='macro'))


train_x, test_x, train_x_raw, test_x_raw, train_y, test_y, label_encoder, vectorizer = parse_file(FILE)
model = train(train_x, train_y)
# pred = predict(model, test_x)
# score(pred, test_y)


# test_utterance = 'ska me sjå kå de me ska disskutere hær ra'
test_utterance = 'da jere nesst\'n ittje eg heller'
print(predict_instance(model, test_utterance, label_encoder, vectorizer))
# print(predict_instance(model, 'ska me sjå kå de me ska disskutere hær ra', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'ska me sjå kå de me ska disskutere hær', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'me sjå kå de me ska disskutere hær ra', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'ska me sjå kå de me ska disskutere her ra', label_encoder, vectorizer)[3].round(2))
# print()
# print(predict_instance(model, 'da jere nesst\'n ittje eg heller', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'da jere nesst\'n ikke eg heller', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'da jere nesst\'n eg heller', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'da jere nesst\'n ittje heller', label_encoder, vectorizer)[3].round(2))
# print(predict_instance(model, 'da nesst\'n ittje eg heller', label_encoder, vectorizer)[3].round(2))

explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform([0, 1, 2, 3]),
                              split_expression='\s+', random_state=42)
labels = [0, 1, 2, 3]
exp = explainer.explain_instance(test_utterance,
                                 lambda z: predict_proba(model, z, vectorizer),
                                 num_features=10, labels=labels)

for label in labels:
    print(label_encoder.inverse_transform([label])[0])
    print('\n'.join(map(str, exp.as_list(label=label))))
    print()
    
# exp.show_in_notebook(text=False)
# exp.show_in_notebook(text=test_utterance, labels=(0,))
# exp.show_in_notebook(text=test_utterance, labels=(1,))
# exp.show_in_notebook(text=test_utterance, labels=(2,))
# exp.show_in_notebook(text=test_utterance, labels=(3,))


c = 0
for _, utterance in test_x_raw.items():
    label = test_y[c]
    interesting_labels = [label]
    pred = predict_instance(model, utterance, label_encoder, vectorizer)
    if pred[0] not in interesting_labels:
        interesting_labels.append(pred[0])
    exp = explainer.explain_instance(utterance,
                                     lambda z: predict_proba(model, z, vectorizer),
                                     num_features=5, labels=interesting_labels)
    
    print('"' + utterance + '""')
    print('ACTUAL', label_encoder.inverse_transform([label])[0])
    print('PREDICTED', pred[1])
    for l in interesting_labels:
        exp.show_in_notebook(text=utterance, labels=(l,))

    c += 1
    if c > 9:
        break
        
    print('\n')


test_utterance = 'eg#vil#ittje'

explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform([0, 1, 2, 3]),
                              split_expression='\s+', random_state=42, bow=False, mask_string='UNK',
                             remove_ngrams=remove_ngrams, utterance2ngrams=utterance2ngrams)
labels = [0, 1, 2, 3]
exp = explainer.explain_instance(test_utterance,
                                 lambda z: predict_proba(model, z, vectorizer),
                                 num_features=10, labels=labels, num_samples=5000)

for label in labels:
    print(label_encoder.inverse_transform([label])[0])
    print('\n'.join(map(str, exp.as_list(label=label))))
    print()
    
# exp.show_in_notebook(text=False)
# exp.show_in_notebook(text=test_utterance, labels=(0,))
# exp.show_in_notebook(text=test_utterance, labels=(1,))
# exp.show_in_notebook(text=test_utterance, labels=(2,))
# exp.show_in_notebook(text=test_utterance, labels=(3,))

