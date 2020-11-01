import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
from lime.lime_text import LimeTextExplainer


np.random.seed(42)


FILE = 'data/phon_parsed.tsv'


def preprocess(utterance):
    return utterance  # TODO


def parse_file(filename, test_size=0.2, max_features=500):
    data = pd.read_csv(filename, encoding='utf8', delimiter='\t',
                       usecols=[0, 3], names=['labels', 'utterances'])
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

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_x = vectorizer.fit_transform(train_x_raw)
    test_x = vectorizer.transform(test_x_raw)

    return train_x, test_x, train_x_raw, test_x_raw, train_y, test_y, label_encoder, vectorizer


def preprocess_and_vectorize(utterance, vectorizer):
    return vectorizer.transform(preprocess(utterance))


def train(train_x, train_y):
    model = svm.LinearSVC(C=1.0)
    model.fit(train_x, train_y)
    return model


def predict(model, test_x):
    return model.predict(test_x)


def predict_instance(model, utterance, label_encoder, vectorizer):
    if isinstance(utterance, str):
        utterance = [utterance]
    x = preprocess_and_vectorize(utterance, vectorizer)
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
        x = preprocess_and_vectorize([utterance], vectorizer)
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
pred = predict(model, test_x)
print(score(pred, test_y))


# test_utterance = 'ska me sjå kå de me ska disskutere hær ra'
test_utterance = 'da jere nesst\'n ittje eg heller'
predict_instance(model, test_utterance, label_encoder, vectorizer)
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



explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform([0, 1, 2, 3]))
labels = [0, 1, 2, 3]
exp = explainer.explain_instance(test_utterance,
                                 lambda z: predict_proba(model, z, vectorizer),
                                 num_features=3, labels=labels)

for label in labels:
    print(label_encoder.inverse_transform([label])[0])
    print('\n'.join(map(str, exp.as_list(label=label))))
    print()
    
exp.show_in_notebook(text=False)
exp.show_in_notebook(text=test_utterance, labels=(0,))
exp.show_in_notebook(text=test_utterance, labels=(1,))
exp.show_in_notebook(text=test_utterance, labels=(2,))
exp.show_in_notebook(text=test_utterance, labels=(3,))


c = 0
for _, utterance in test_x_raw.items():
    label = test_y[c]
    interesting_labels = [label]
    pred = predict_instance(model, utterance, label_encoder, vectorizer)
    if pred[0] not in interesting_labels:
        interesting_labels.append(pred[0])
    exp = explainer.explain_instance(utterance,
                                     lambda z: predict_proba(model, z, vectorizer),
                                     num_features=3, labels=interesting_labels)
    
    print('"' + utterance + '""')
    print('ACTUAL', label_encoder.inverse_transform([label])[0])
    print('PREDICTED', pred[1])
    for l in interesting_labels:
        exp.show_in_notebook(text=utterance, labels=(l,))

    c += 1
    if c > 9:
        break
        
    print('\n')

