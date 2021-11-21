import tensorflow as tf

tf.__version__

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from board_operations import *

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1*1*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((1, 1, 128)))
    assert model.output_shape == (None, 1, 1, 128)

    model.add(layers.Conv2DTranspose(128, (5, 5)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Softmax())
    #model.add(layers.Softmax())

    assert model.output_shape == (None, 7, 7, NUMBER_OF_TILE_TYPES)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same', input_shape=[7, 7, NUMBER_OF_TILE_TYPES]))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LayerNormalization())
    #model.add(layers.BatchNormalization()) # SWITCHED TO LAYERNORM SINCE WGAN DOES NOT ACCEPT BATCHNORM
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LayerNormalization())
    #model.add(layers.BatchNormalization()) # SWITCHED TO LAYERNORM SINCE WGAN DOES NOT ACCEPT BATCHNORM
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LayerNormalization())
    #model.add(layers.BatchNormalization()) # SWITCHED TO LAYERNORM SINCE WGAN DOES NOT ACCEPT BATCHNORM
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LayerNormalization())
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LayerNormalization())
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (5, 5)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())

    model.add(layers.Dense(1))
    #model.add(layers.Activation(activation="sigmoid"))   
    #model.add(layers.Activation(activation="sigmoid"))   
    #model.add(layers.LeakyReLU(alpha=0.2))

#    model.add(layers.Flatten())
#
#    model.add(layers.Dense(32))
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU(alpha=0.2))
#    model.add(layers.Dense(1))

    return model

checkpoint_dir = './training_checkpoints'

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

print(tf.train.latest_checkpoint(checkpoint_dir))

noise_dim = 100
num_examples_to_generate = 1000

seed = tf.random.normal([num_examples_to_generate, noise_dim])

def test_model(model):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(seed, training=False)

    #print(predictions[0])

    room_list = []
    valid_rooms = 0

    for i in range(predictions.shape[0]):
        room = np.expand_dims(predictions[i], 0)
        room = transform_from_one_hot(room)
        room_list.append(room)
        if(valid_room(room)):
#            print("valid")
            valid_rooms += 1
            print(np.reshape(room, (7, 7)))

    unique_rooms_num = len(get_unique(room_list))

    valid_percent = (float(valid_rooms) / num_examples_to_generate)
    valid_unique_percent = (float(valid_rooms) / unique_rooms_num)
    print("Generated " + str(valid_percent*100) + " prcnt of valid rooms from which " + str(valid_unique_percent*100) + " prcnt are unique") 

test_model(generator)