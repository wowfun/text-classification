import os
import time
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from utils import data_helper, result_helper
from models.lstm_model import LSTMModel, BiLSTM

# TODO: 1. NOTE: 训练集、测试集在向量化时，必须一起同时向量化，否则两者的相同词语的词向量的表示会不一致，导致测试集的分类错误
# 2. callbacks move to base model


class Args:
    def __init__(self, model_name='lstm'):
        self.model_name = model_name
        self.modes = ['train', 'pred']  #

        # 数据
        self.train_dataset = 'data/train/train_processed.csv'
        self.test_dataset = 'data/test_data_processed.csv'
        self.tokenizer_path = 'data/tokenizer-10.pickle'
        self.val_split = 0.1
        self.max_num_words = 100000  # 最多保留不同词语词数，频率从大到小
        self.max_sequence_len = 256  # = input_len
        self.num_labels = 10  # = output_units

        # 模型超参
        self.epchos = 60
        self.batch_size = 64
        self.embedding_dims = 128


    @property
    def checkpoint_dir(self):
        dir_path = "./saved_models/{}/".format(self.model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir + "/{}.ckpt".format(time.strftime("%Y%m%d"))

    @property
    def h5_path(self): # 不使用checkpoint保存模型，使用.h5保存
        return "./saved_models/{}.h5".format(self.model_name)



class ResultArgs:
    def __init__(self):
        self.save_path = 'results/result.csv'
        self.class_label_to_class_dict = {
            0: '家居', 1: '房产', 2: '教育', 3: '时尚', 4: '时政', 5: '科技', 6: '财经', 7: '游戏', 8: '娱乐', 9: '体育'}
        self.rank_label_to_rank_dict = {0: '可公开', 1: '低风险', 2: '中风险', 3: '高风险'}
        self.class_label_to_rank_label_dict = {
            0: 0, 1: 2, 2: 1, 3: 1, 4: 3, 5: 2, 6: 3, 7: 1, 8: 0, 9: 0}

        @property
        def class_to_class_label_dict(self):
            return {val: key for key, val in self.class_label_to_class_dict.items()}


if __name__ == "__main__":
    args = Args('bilstm')

    # 处理 测试集 原数据
    # X_pred = pd.read_csv(args.dataset_2)
    # X_pred['cutted_content'] = data_helper.preprocess_text_series(
    #     X_pred['content'])
    # X_pred.to_csv('data/test_data_processed.csv', index=False)

    # build
    model1 = LSTMModel(args.checkpoint_path)
    # model1=BiLSTM(args.checkpoint_path)

    model1.build(input_len=args.max_sequence_len,
                 lstm_units=args.max_sequence_len, output_units=args.num_labels)

    # train or load
    if 'train' in args.modes:
        print('* Training')
        X, Y = data_helper.load_preprocessed_data_from_csv(
            args.train_dataset, with_Y=True, onehot=True)
        tokenizer_mode = 'load'
        if not os.path.exists(args.tokenizer_path):
            tokenizer_mode = 'create'
        X, _ = data_helper.tokenize(
            lang=X, mode=tokenizer_mode, path=args.tokenizer_path, max_num_words=args.max_num_words,  max_sequence_len=args.max_sequence_len)

        print('* X shape ', X.shape)
        print('* Y shape ', Y.shape)

        history = model1.train(
            X, Y, epochs=args.epchos, batch_size=args.batch_size, val_split=args.val_split)
        result_helper.save_metrics(history)
        model1.model.save(args.h5_path) # h5
    else:
        model1.load(args.checkpoint_dir)
        # model1 = tf.keras.models.load_model(args.h5_path) # load h5

    # pred
    if 'pred' in args.modes:
        print('* Predicting')
        res_args = ResultArgs()

        X_test = data_helper.load_preprocessed_data_from_csv(
            args.test_dataset, with_Y=False)
        X_test, _ = data_helper.tokenize(
            X_test, mode='load', path=args.tokenizer_path, max_num_words=args.max_num_words, max_sequence_len=args.max_sequence_len)
        preds, _ = model1.pred(X_test)
        print('* X for pred shape ', X_test.shape)
        print('* preds shape ', preds.shape)
        test_df = pd.read_csv(args.test_dataset)
        id_series = test_df['id']
        result_helper.results_to_submit(preds, save_path=res_args.save_path,
                                        id_series=id_series,
                                        class_label_to_class_dict=res_args.class_label_to_class_dict,
                                        rank_label_to_rank_dict=res_args.rank_label_to_rank_dict,
                                        class_label_to_rank_label_dict=res_args.class_label_to_rank_label_dict)
