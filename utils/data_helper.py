import re
import pickle

import pandas as pd
import tensorflow as tf

import jieba as jb


def remove_punctuation(line):  # 删除除字母,数字，汉字以外的所有符号
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def get_stopwords_list(filepath):  # 获取停用词
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def load_preprocessed_data_from_csv(dataset, with_Y=True, onehot=True):
    dataset_df = pd.read_csv(dataset)
    X = dataset_df['cutted_content'].values
    if with_Y:
        if onehot:
            Y = pd.get_dummies(dataset_df['label']).values
        else:
            Y = dataset_df['label'].values
        return X, Y
    return X


def tokenize(lang, mode='load', path=None, max_num_words=None, max_sequence_len=256):  # mode: create or load
    if mode == 'load':
        with open(path, 'rb') as handle:
            lang_tokenizer = pickle.load(handle)
        print('** Load tokenzier from: ', path)
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words,
                                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
        lang_tokenizer.fit_on_texts(lang)
        # saving
        with open(path, 'wb') as handle:
            pickle.dump(lang_tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('** Save tokenizer at: ', path)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_sequence_len,
                                                           padding='post', truncating='post')  # NOTE
    print('** Total different words: %s.' % len(lang_tokenizer.word_index))

    return tensor, lang_tokenizer


def preprocess_text_series(text_series, stopwords_path=None):
    cleaned_texts = text_series.apply(remove_punctuation)
    stopwords = []
    if stopwords_path != None:
        stopwords = get_stopwords_list(stopwords_path)
    cutted_texts = cleaned_texts.apply(lambda x: " ".join(
        [w for w in list(jb.cut(x)) if w not in stopwords]))
    return cutted_texts
