import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # 禁用gpu
import numpy as np
import tensorflow as tf


class BaseModel:
    def __init__(self, model=None, loss='categorical_crossentropy', metrics=['accuracy'], checkpoint_path=None, callbacks=[]):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.checkpoint_path = checkpoint_path
        if callbacks != []:
            self.callbacks = callbacks
        else:
            self.set_callbacks()

    def build(self):
        pass

    def set_callbacks(self):
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, monitor='val_loss', mode='auto', save_best_only=True, save_weights_only=True, verbose=1, save_freq='epoch')
        early_stopping_cb=tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3
        )
        self.callbacks = []
        self.callbacks.append(save_model_cb)
        self.callbacks.append(early_stopping_cb)
        print('*** callback nums ',len(self.callbacks))

    def train(self, input, target=None, epochs=20, batch_size=64, val_split=0.1):
        self.model.compile(loss=self.loss,
                           optimizer='adam',
                           metrics=self.metrics)

        history = self.model.fit(x=input, y=target, epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=val_split,
                                 callbacks=self.callbacks)

        return history

    def test(self):
        pass

    def pred(self, input, with_arg=True):
        probs = self.model.predict(input)
        if not with_arg:
            return probs
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def load(self, checkpoint_path, latest=True):
        if os.path.isdir(checkpoint_path) and latest:
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            print("** load latest ckpt file: ", checkpoint_path)
        self.model.load_weights(checkpoint_path)
        return self.model
