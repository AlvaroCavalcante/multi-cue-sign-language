from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class Attention(Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):

        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x, self.W)+self.b) # gerar os pesos das sequência, sendo 1 peso por time-step (frame)
        a = K.softmax(e, axis=1) # coloca os pesos entre 0 e 1
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
