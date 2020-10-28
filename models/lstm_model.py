import tensorflow as tf

from .base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self,checkpoint_path=None):
        super().__init__(model=None, loss='categorical_crossentropy',
                         metrics=['accuracy'], callbacks=[])

        self.checkpoint_path=checkpoint_path
        self.set_callbacks()

    def build(self, input_len, output_units, max_num_words=100000, embedding_dims=128, lstm_units=128, dropout=0.2, recurrent_dropout=0.2):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                max_num_words, embedding_dims, input_length=input_len),
            tf.keras.layers.SpatialDropout1D(dropout),
            tf.keras.layers.LSTM(lstm_units, dropout=dropout,
                                 recurrent_dropout=recurrent_dropout),
            tf.keras.layers.Dense(output_units, activation='softmax'),

        ])
        print(self.model.summary())
        return self.model

    def set_callbacks(self):
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, monitor='val_loss',    mode='auto',  # save_best_only=True,
                                                           save_weights_only=True, verbose=1,
                                                           save_freq='epoch')

        self.callbacks = []
        self.callbacks.append(save_model_cb)
