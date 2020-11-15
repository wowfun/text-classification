import tensorflow as tf
from tensorflow.keras import regularizers

from .base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, checkpoint_path=None):
        super().__init__(model=None, loss='categorical_crossentropy',
                         metrics=['accuracy'], checkpoint_path=checkpoint_path, callbacks=[])

    def build(self, input_len, output_units, max_num_words=100000, embedding_dims=128, lstm_units=128, dropout=0.1, recurrent_dropout=0.1,regularizer_factor=0.001):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                max_num_words, embedding_dims, input_length=input_len),
            tf.keras.layers.SpatialDropout1D(dropout),
            tf.keras.layers.LSTM(lstm_units, dropout=dropout,
                                 recurrent_dropout=recurrent_dropout, kernel_regularizer=regularizers.l2(regularizer_factor)),
            tf.keras.layers.Dense(
                int(lstm_units/2), activation='relu', kernel_regularizer=regularizers.l2(regularizer_factor)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(output_units, activation='softmax'),
        ])
        print(self.model.summary())
        return self.model


class BiLSTM(LSTMModel):
    def __init__(self, checkpoint_path=None):
        super().__init__(checkpoint_path=checkpoint_path)

    def build(self, input_len, output_units, max_num_words=100000, embedding_dims=128, lstm_units=128, dropout=0.1, recurrent_dropout=0.2, regularizer_factor=0.001):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                max_num_words, embedding_dims, input_length=input_len),
            tf.keras.layers.SpatialDropout1D(dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(regularizer_factor))),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                int(lstm_units/2), kernel_regularizer=regularizers.l2(regularizer_factor))),
            tf.keras.layers.Dense(
                int(lstm_units/4), activation='relu', kernel_regularizer=regularizers.l2(regularizer_factor)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(output_units, activation='softmax'),

        ])
        print(self.model.summary())
        return self.model
