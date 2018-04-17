from abc import abstractmethod, ABC

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed
from keras import backend as K
from keras.models import Model

import numpy as np


class LanguageModel(Model):
    def __init__(self, config):
        self.question = Input(shape=(config['question_len'],), dtype='int32', name='question_base')
        self.answer_good = Input(shape=(config['answer_len'],), dtype='int32', name='answer_good_base')
        self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32', name='answer_bad_base')

        self.config = config
        self.params = config.get('similarity', dict())

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._answer = None
        self._qa_model = None

        self.training_model = None
        self.prediction_model = None

    def get_answer(self):
        if self._answer is None:
            self._answer = Input(shape=(self.config['answer_len'],), dtype='int32', name='answer')
        return self._answer

    @abstractmethod
    def build(self):
        return

    def get_similarity(self):
        ''' Specify similarity in configuration under 'similarity' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity'

        Example configuration:

        config = {
            ... other parameters ...
            'similarity': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }

        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.params
        similarity = params['mode']

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

        if similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_qa_model(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model is None:
            question_output, answer_output = self._models
            dropout = Dropout(self.params.get('dropout', 0.2))
            similarity = self.get_similarity()
            # qa_model = merge([dropout(question_output), dropout(answer_output)],
            #                  mode=similarity, output_shape=lambda _: (None, 1))
            qa_model = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(question_output),
                                                                             dropout(answer_output)])
            self._qa_model = Model(inputs=[self.question, self.get_answer()], outputs=qa_model, name='qa_model')

        return self._qa_model

    def compile(self, optimizer, **kwargs):
        qa_model = self.get_qa_model()

        good_similarity = qa_model([self.question, self.answer_good])
        bad_similarity = qa_model([self.question, self.answer_bad])

        # loss = merge([good_similarity, bad_similarity],
        #              mode=lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
        #              output_shape=lambda x: x[0])

        loss = Lambda(lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
                      output_shape=lambda x: x[0])([good_similarity, bad_similarity])

        self.prediction_model = Model(inputs=[self.question, self.answer_good], outputs=good_similarity,
                                      name='prediction_model')
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

        self.training_model = Model(inputs=[self.question, self.answer_good, self.answer_bad], outputs=loss,
                                    name='training_model')
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=(x[0].shape[0],)) # doesn't get used
        return self.training_model.fit(x, y, **kwargs)

    def predict(self, x):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        return self.prediction_model.predict_on_batch(x)

    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)
