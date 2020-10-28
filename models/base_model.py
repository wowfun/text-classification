import tensorflow as tf


class BaseModel:
    def __init__(self, model=None, loss='categorical_crossentropy', metrics=['accuracy'], callbacks=[]):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks

    def build(self):
        pass

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

    def pred(self):
        pass
