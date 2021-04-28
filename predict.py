# coding: utf-8

# called by predict_fold.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ngram_lime.lime.lime_text import LimeTextExplainer
import datetime
import torch
from keras.models import Model
from keras.layers import Bidirectional, Dense, Dropout, Input, GRU, Activation
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def split_ngrams(joined_ngrams):
    # Used by the TfidfVectorizer, important for the modified LIME
    return joined_ngrams.split(' ')


def encode(ngrams_train, ngrams_test, labels_train, labels_test,
           max_features=5000):

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(labels_train)
    test_y = label_encoder.transform(labels_test)

    vectorizer = TfidfVectorizer(max_features=max_features,
                                 analyzer=split_ngrams)
    train_x = vectorizer.fit_transform(ngrams_train)
    test_x = vectorizer.transform(ngrams_test)

    return train_x, test_x, train_y, test_y, label_encoder, vectorizer


def get_embeddings(utterances, max_len, embedding_size, batch_size,
                   flaubert_tokenizer, flaubert, flatten):
    token_ids = [flaubert_tokenizer.encode(utterance, max_length=max_len,
                                           truncation=True,
                                           padding='max_length')
                 for utterance in utterances]
    n = len(token_ids)
    x = torch.empty((n, max_len, embedding_size))
    print(n, batch_size)
    for i in range(0, n, batch_size):
        print("- Batch {}--{}".format(i, i + batch_size))
        batch = flaubert(torch.tensor(token_ids[i:i + batch_size]))[0]
        x[i:i + batch_size] = batch
    if flatten:
        x = torch.flatten(x, start_dim=1)
    x = x.detach().numpy()
    if flatten:
        assert x.shape == (n, max_len * embedding_size)
    return x


def encode_embeddings(toks_train, toks_test, labels_train, labels_test,
                      flaubert_tokenizer, flaubert, max_len,
                      batch_size, embedding_size, flatten):
    train_x = get_embeddings(toks_train, max_len, embedding_size, batch_size,
                             flaubert_tokenizer, flaubert, flatten)
    test_x = get_embeddings(toks_test, max_len, embedding_size, batch_size,
                            flaubert_tokenizer, flaubert, flatten)
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(labels_train)
    test_y = label_encoder.transform(labels_test)
    return train_x, test_x, train_y, test_y, label_encoder


def preprocess_and_vectorize(utterance, vectorizer):
    return vectorizer.transform([utterance])
    # return vectorizer.transform([preprocess(utterance)])


def train(train_x, train_y, model_type, n_classes, linear_svc, log_file,
          learning_rate, dropout_rate, class_weight, verbose, hidden_size,
          epochs, batch_size):
    if 'nn' in model_type:
        return train_gru(train_x, train_y, 'attn' in model_type, n_classes,
                         class_weight, log_file, hidden_size, epochs,
                         batch_size, learning_rate, dropout_rate, verbose)

    if linear_svc:
        model = svm.LinearSVC(C=1.0, class_weight=class_weight,
                              verbose=verbose)
    else:
        # Binary cases
        model = svm.SVC(C=1.0, probability=True, class_weight=class_weight,
                        verbose=verbose)
    model.fit(train_x, train_y)
    return model


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.Q = Dense(self.hidden_size)
        self.softmax = Activation('softmax')

    def call(self, x):
        dot_similarity = self.Q(x) / (self.hidden_size ** 0.5)
        attn = self.softmax(dot_similarity)
        out = x * attn
        return attn, K.sum(out, axis=1)


def train_gru(train_x, train_y, attention_layer, n_classes, class_weight,
              log_file, hidden_size, epochs, batch_size, learning_rate,
              dropout_rate, verbose,
              loss='sparse_categorical_crossentropy',
              metric='categorical_accuracy'):
    n_timesteps, embed_depth = train_x.shape[-2], train_x.shape[-1]

    inputs = Input(shape=(n_timesteps, embed_depth), name='inputs')

    gru = Bidirectional(GRU(hidden_size, return_sequences=attention_layer,
                            return_state=False, name='gru'),
                        name='bidi_gru')
    # gru_out, gru_fwd_state, gru_bwd_state = gru(inputs)
    gru_out = gru(inputs)

    dropout = Dropout(dropout_rate, name='dropout')
    dropout_out = dropout(gru_out)

    if attention_layer:
        attention = Attention()
        attn_scores, attn_out = attention(dropout_out)
        dense_in = attn_out
    else:
        dense_in = dropout_out

    dense = Dense(n_classes, activation='softmax', name='softmax')
    dense_out = dense(dense_in)

    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=inputs, outputs=dense_out)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.summary(line_length=100)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        class_weight=class_weight, verbose=1)

    if attention_layer:
        # Is exactly the same as the model used for training, except that it
        # also outputs the attention scores.
        # Its weights are the weights learned by fitting the other model!
        attn_model = Model(inputs=inputs,
                           outputs=[dense_out, attn_scores])

    with open(log_file, 'a', encoding='utf8') as f:
        if attention_layer:
            attn_model.summary(print_fn=lambda x: f.write(x + '\n'),
                               line_length=100)
        else:
            model.summary(print_fn=lambda x: f.write(x + '\n'),
                          line_length=100)
        f.write('LOSS {}\n'.format(loss))
        f.write(str(history.history['loss']) + '\n')
        f.write('METRIC {}\n'.format(metric))
        f.write(str(history.history[metric]) + '\n')

    if attention_layer:
        return attn_model
    return model


def predict_instance(model, utterance, label_encoder, vectorizer, linear_svc):
    # x = vectorizer.transform([preprocess(utterance)])
    x = vectorizer.transform([utterance])
    pred = model.predict(x)
    margins = model.decision_function(x)
    if linear_svc:
        exp = np.exp(margins)
        softmax = exp / np.sum(exp)
    #     print("{}: {}".format(utterance, softmax.round(3)))
        return (pred[0], label_encoder.inverse_transform(pred)[0],
                margins, softmax)
    return (pred[0], label_encoder.inverse_transform(pred)[0],
            margins, model.predict_proba(x))


def predict_proba(model, data, vectorizer, linear_svc, n_labels):
    probs = np.zeros((len(data), n_labels))
    for i, utterance in enumerate(data):
        x = vectorizer.transform([utterance])
        if linear_svc:
            # pred = model.predict(x)
            margins = model.decision_function(x)
            exp = np.exp(margins)
            probs[i] = exp / np.sum(exp)  # softmax
        else:
            probs[i] = model.predict_proba(x)
    return probs


def predict_proba_embeddings(model, data, flaubert, flaubert_tokenizer,
                             linear_svc, neural, max_len, n_labels):
    probs = np.zeros((len(data), n_labels))
    for i, utterance in enumerate(data):
        token_ids = [flaubert_tokenizer.encode(utterance, max_length=max_len,
                                               truncation=True,
                                               padding='max_length')]
        x = flaubert(torch.tensor(token_ids))[0]
        if neural:
            probs[i] = model.predict(x.detach().cpu().numpy())
        elif linear_svc:
            # pred = model.predict(x)
            margins = model.decision_function(x[0])
            exp = np.exp(margins)
            probs[i] = exp / np.sum(exp)  # softmax
        else:
            probs[i] = model.predict_proba(x[0])
    return probs


def predict_proba_lstm(model, data, flaubert, flaubert_tokenizer,
                       linear_svc, neural, max_len, n_labels):
    # probs = np.zeros((len(data), n_labels))
    token_ids = [flaubert_tokenizer.encode(utterance, max_length=max_len,
                                           truncation=True,
                                           padding='max_length')
                 for utterance in data]
    x = flaubert(torch.tensor(token_ids))[0]
    probs = model.predict(x.detach().cpu().numpy())
    print(len(data))
    print(probs.shape)
    return probs


def score(pred, test_y):
    return (accuracy_score(test_y, pred),
            f1_score(test_y, pred, average='macro'),
            confusion_matrix(test_y, pred))


def support_vectors(model, label_encoder, train_x, train_x_raw,
                    labels=[0, 1, 2, 3]):
    for label in labels:
        print(label_encoder.inverse_transform([label])[0])
        print('==========================\n')
        dec_fn = model.decision_function(train_x)[:, label]
        # support vectors and vectors 'in the middle of the street'
        support_vector_indices_pos = np.where(
            np.logical_and(dec_fn > 0, dec_fn <= 1))[0]
        print('positive side')
        for idx in support_vector_indices_pos:
            print('-', train_x_raw[idx])
        support_vector_indices_neg = np.where(
            np.logical_and(dec_fn <= 0, dec_fn >= -1))[0]
        print()
        print('negative side')
        for idx in support_vector_indices_neg:
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


def explain_lime(classifier, vectorizer, label_encoder, n_labels, test_x_raw,
                 test_x_ngrams, test_x, test_y, out_folder, n_lime_features,
                 num_lime_samples, linear_svc, neural, recalculate_ngrams,
                 flaubert=None, flaubert_tokenizer=None,
                 max_len=None):
    labels = list(range(n_labels))
    explainer = LimeTextExplainer(class_names=label_encoder.inverse_transform(
                                    labels),
                                  split_expression=split_ngrams,
                                  bow=True, ngram_lvl=True,
                                  utterance2ngrams=split_ngrams,
                                  recalculate_ngrams=recalculate_ngrams)
    if neural:
        predict_function = lambda z: predict_proba_lstm(
            classifier, z, flaubert, flaubert_tokenizer,
            linear_svc, neural, max_len, n_labels)
    elif flaubert:
        predict_function = lambda z: predict_proba_embeddings(
            classifier, z, flaubert, flaubert_tokenizer,
            linear_svc, neural, max_len, n_labels)
    else:
        predict_function = lambda z: predict_proba(classifier, z, vectorizer,
                                                   linear_svc, n_labels)

    for lab_raw in label_encoder.inverse_transform(labels):
        with open('{}/importance_values_{}.txt'.format(out_folder, lab_raw),
                  'w+', encoding='utf8') as f:
            # Making sure we have new files instead of potentially adding onto
            # old runs.
            pass
    with open(out_folder + 'predictions.tsv', 'w+', encoding='utf8') as f_pred:
        for idx, (utterance, ngrams, encoded, y) in enumerate(
                zip(test_x_raw, test_x_ngrams, test_x, test_y)):
            y_raw = label_encoder.inverse_transform([y])[0]
            if neural:
                pred_enc = np.argmax(classifier.predict(np.array([encoded])))
            else:
                pred_enc = classifier.predict(encoded)[0]
            pred_raw = label_encoder.inverse_transform([pred_enc])[0]
            f_pred.write('{}\t{}\t{}\t{}\n'.format(idx, utterance,
                                                   y_raw, pred_raw))

            exp = explainer.explain_instance(ngrams,
                                             predict_function,
                                             num_features=n_lime_features,
                                             labels=labels,
                                             num_samples=num_lime_samples
                                             )
            for lab in labels:
                lime_results = exp.as_list(label=lab)
                lab_raw = label_encoder.inverse_transform([lab])[0]
                prediction_score = exp.score[lab]
                with open('{}/importance_values_{}.txt'.format(out_folder,
                                                               lab_raw),
                          'a', encoding='utf8') as f:
                    for feature, coeff in lime_results:
                        f.write('{}\t{}\t{:.10f}\t{:.10f}\n'
                                .format(idx, feature, coeff, prediction_score))

            if idx % 50 == 0:
                now = datetime.datetime.now()
                print(idx)
                print(now)
                print('"' + utterance + '""')
                print('ACTUAL', y_raw)
                print('PREDICTED', pred_raw)
                print('\n')
                with open(out_folder + '/log.txt', 'a', encoding='utf8') as f:
                    f.write('{} -- {} -- "{}"\n'.format(idx, now, utterance))
                    f.write('ACTUAL: ' + str(y_raw) + '\n')
                    f.write('PREDICTED: ' + str(pred_raw) + '\n')
                    for label_nr in range(n_labels):
                        lab = label_encoder.inverse_transform([label_nr])[0]
                        f.write('Class ' + lab + ': ' + ', '.join(
                            ['{}\t{:.5f}'.format(x[0], x[1])
                             for x in exp.as_list(label=label_nr)[:5]]) + '\n')
                    f.write('\n')
