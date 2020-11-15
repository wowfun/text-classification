import tensorflow as tf
from tensorflow.keras import regularizers

from .base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, checkpoint_path=None):
        super().__init__(model=None, loss='categorical_crossentropy',
                         metrics=['accuracy'], checkpoint_path=checkpoint_path, callbacks=[])

    def build(self, input_len, output_units, max_num_words=100000, embedding_dims=128, lstm_units=128, dropout=0.2, recurrent_dropout=0.2):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                max_num_words, embedding_dims, input_length=input_len),
            tf.keras.layers.SpatialDropout1D(dropout),
            tf.keras.layers.LSTM(lstm_units, dropout=dropout,
                                 recurrent_dropout=recurrent_dropout, kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dense(output_units, activation='softmax'),
        ])
        print(self.model.summary())
        return self.model


class BiLSTM(LSTMModel):
    def __init__(self, checkpoint_path=None):
        super().__init__(checkpoint_path=checkpoint_path)

    def build(self, input_len, output_units, max_num_words=100000, embedding_dims=128, lstm_units=128, dropout=0.2, recurrent_dropout=0.2):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                max_num_words, embedding_dims, input_length=input_len),
            tf.keras.layers.SpatialDropout1D(dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                lstm_units, kernel_regularizer=regularizers.l2(0.001))),
            tf.keras.layers.Dense(
                output_units, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dense(output_units, activation='softmax'),

        ])
        print(self.model.summary())
        return self.model
