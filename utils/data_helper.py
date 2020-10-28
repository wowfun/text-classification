import pandas as pd
import tensorflow as tf


def load_data_from_csv(dataset, onehot=True):
    dataset_df = pd.read_csv(dataset)
    X = dataset_df['cuted_content'].values
    if onehot:
        Y = pd.get_dummies(dataset_df['label']).values
    else:
        Y = dataset_df['label'].values
    return X, Y


def tokenize(lang, max_num_words=None, max_sequence_len=300):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words,
                                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_sequence_len,
                                                           padding='post', truncating='post')  # NOTE
    print('** Total different words: %s.' % len(lang_tokenizer.word_index))
    return tensor, lang_tokenizer
