params = {
    'max_len': 15,
    'hidden_size': 256,
    'embedding_dim': 300,
    'batch_size': 300,
    'n_epochs': 1000,
    'N': 5
}

locations = {
    'train_loc': 'data/train',
    'dev_loc': 'data/dev',
    'fasttext_loc': 'vocabulary/crawl-300d-2M.vec',
    'w2vec_loc': 'vocabulary/word_to_vec.pkl',
    'model_loc': 'models/model.pt',
    'stopwordsfile': 'data/stop_words.txt'
}