import tensorflow as tf 
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

def create_mlp(input_size, output_size = 3, layers = [128, 128], activation = "relu"):

    inputs = Input((input_size, ))
    for i, l in enumerate(layers):
        if i == 0:
            x = Dense(l, activation = activation)(inputs)
        else:
            x = Dense(l, activation = activation)(x)
    outputs = Dense(output_size, activation = "linear")(x)
    model = Model(inputs, outputs)
    return model

