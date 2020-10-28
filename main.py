import os
import time

import matplotlib.pyplot as plt

from utils import data_helper
from models.lstm_model import LSTMModel

class Args:
    def __init__(self,model_name='lstm'):
        self.model_name=model_name
        self.modes=['train','test','pred']
        self.dataset='data/train/labeled_data_processed.csv'

        # 文本向量
        self.max_num_words=100000 # 最多保留不同词语词数，频率从大到小
        self.max_sequence_len=256

        # 模型超参
        self.embedding_dims=128

    @property
    def checkpoint_dir(self):
        dir_path="./saved_models/{}/".format(self.model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir + "/{}.ckpt".format(time.strftime("%Y%m%d"))


if __name__ == "__main__":
    args=Args()
    X,Y=data_helper.load_data_from_csv(args.dataset)
    print('* X shape ',X.shape)
    print('* Y shape ',Y.shape)
    X,_=data_helper.tokenize(X,args.max_num_words,args.max_sequence_len)
    print('* X shape ',X.shape)
    model1=LSTMModel(args.checkpoint_path)
    model1.build(input_len=X.shape[1],output_units=Y.shape[1])
    history=model1.train(X,Y)


    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plt.show()
    plt.savefig('loss.png')

    plt.figure()
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.legend()
    # plt.show()
    plt.savefig('acc.png')

    

