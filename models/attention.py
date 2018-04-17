from models.language_model import LanguageModel

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed
from keras import backend as K
from keras.models import Model
import numpy as np 

class AttentionModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              # mask_zero=True,
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # question rnn part
        f_rnn = LSTM(141, return_sequences=True, consume_less='mem')
        b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)
        question_f_rnn = f_rnn(question_embedding)
        question_b_rnn = b_rnn(question_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = merge([maxpool(question_f_rnn), maxpool(question_b_rnn)], mode='concat', concat_axis=-1)

        # answer rnn part
        from models.attention_lstm import AttentionLSTMWrapper
        f_rnn = AttentionLSTMWrapper(f_rnn, question_pool, single_attention_param=True)
        b_rnn = AttentionLSTMWrapper(b_rnn, question_pool, single_attention_param=True)

        answer_f_rnn = f_rnn(answer_embedding)
        answer_b_rnn = b_rnn(answer_embedding)
        answer_pool = merge([maxpool(answer_f_rnn), maxpool(answer_b_rnn)], mode='concat', concat_axis=-1)

        return question_pool, answer_pool
