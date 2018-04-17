from models.language_model import LanguageModel

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed
from keras import backend as K
from keras.models import Model

import numpy as np

class EmbeddingModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              mask_zero=True,
                              # dropout=0.2,
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_embedding)
        answer_pool = maxpool(answer_embedding)

        return question_pool, answer_pool