from models.embedding_model import EmbeddingModel

conf = {
        'model_name' : 'EmbeddingModel',
        'model' : EmbeddingModel,
        'n_words': 22353,
        'question_len': 150,
        'answer_len': 150,
        'margin': 0.009,
        'initial_embed_weights': 'word2vec_100_dim.embeddings',
        'early_stopping_patience' : 10,

        'training': {
            'batch_size': 100,
            'nb_epoch': 300,
            'validation_split': 0.1,
        },

        'similarity': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
            'dropout': 0.5,
        }
    }