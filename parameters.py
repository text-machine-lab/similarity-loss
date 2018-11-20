params = {
    'max_len': 15,
    'hidden_size': 256,
    'embedding_dim': 300,
    'batch_size': 10,
    'n_epochs': 1000,
    'N': 5,     # For soft-label loss only
    'bigrams': False
}

locations = {
    'train_loc': 'data/train',
    'test_loc': 'data/dev',
    'embeddings_loc': 'vocabulary/crawl-300d-2M.vec',
    'w2vec_loc': 'vocabulary/word_to_vec.pkl',
    'model_loc': 'models/model.pt',
    'stopwordsfile': 'data/stop_words.txt'
}