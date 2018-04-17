from models.language_model import LanguageModel

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed
from keras import backend as K
from keras.models import Model

import numpy as np

class ConvolutionModel(LanguageModel):
    def build(self):
        assert self.config['question_len'] == self.config['answer_len']

        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        hidden_layer = TimeDistributed(Dense(200, activation='tanh'))

        question_hl = hidden_layer(question_embedding)
        answer_hl = hidden_layer(answer_embedding)

        # cnn
        cnns = [Conv1D(kernel_size=kernel_size,
                       filters=1000,
                       activation='tanh',
                       padding='same') for kernel_size in [2, 3, 5, 7]]
        # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
        question_cnn = concatenate([cnn(question_hl) for cnn in cnns], axis=-1)
        # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
        answer_cnn = concatenate([cnn(answer_hl) for cnn in cnns], axis=-1)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        # enc = Dense(100, activation='tanh')
        # question_pool = enc(maxpool(question_cnn))
        # answer_pool = enc(maxpool(answer_cnn))
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        return question_pool, answer_pool