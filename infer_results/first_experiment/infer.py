from operator import ge
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import sigmoid
from tensorflow.python.keras.layers.core import Activation, Flatten

tf.__version__

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from board_operations import *

from image_creator import valid_room

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



    return model



#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise_dim = 100
num_examples_to_generate = 10000

ACCEPTANCE_UNIQUE_THRESHOLD = 75.0

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# to compare indexes as ints instead of strings
def custom_files_compare(filename1):#, filename2):
    checkpoint_number1 = int(filename1[5:])
    return checkpoint_number1

def get_valid_checkpoints(files):
    valid_checkpoints = []

    for file in files:
        if file.endswith(".index"):
            valid_checkpoints.append(file[0:-6])

    return valid_checkpoints

def test_checkpoints():
    checkpoint_dir = './training_checkpoints'
    checkpoints_files = get_valid_checkpoints(os.listdir(checkpoint_dir))
    max_valid_prcnt = 0.0
    max_path = ""

    checkpoints_files.sort(reverse=False, key=custom_files_compare)

    valid_values_list = []
    unique_values_list = []

    for checkpoint_file in checkpoints_files:
        filepath = os.path.join(checkpoint_dir, checkpoint_file)
            
        generator = make_generator_model()
        discriminator = make_discriminator_model()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                            discriminator_optimizer=discriminator_optimizer,
                                            generator=generator,
                                            discriminator=discriminator)
                                            

        checkpoint.restore(filepath).expect_partial()

        print("Testing " + filepath)
        valid_prcnt, unique_prcnt = test_model(generator)

        valid_values_list.append(valid_prcnt)
        unique_values_list.append(unique_prcnt)

        if max_valid_prcnt < valid_prcnt and unique_prcnt > ACCEPTANCE_UNIQUE_THRESHOLD:
            max_valid_prcnt = valid_prcnt
            max_path = filepath

    print("Finished testing")
    print(max_valid_prcnt)
    print(max_path)

    with open("checkpoints_testing.txt", "w") as file:
        for i in range(len(valid_values_list)):
            file.write(str(valid_values_list[i]) + "\n")
            file.write(str(unique_values_list[i]) + "\n")

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
        if(valid_room(room)):
            valid_rooms += 1
            room_list.append(room)

    unique_rooms_num = len(get_unique(room_list))

    valid_percent = (float(valid_rooms) / num_examples_to_generate) * 100

    if valid_rooms != 0:
        valid_unique_percent = (unique_rooms_num / float(valid_rooms)) * 100
    else:
        valid_unique_percent = 0

    print("Generated " + str(valid_percent) + " prcnt of valid rooms from which " + str(valid_unique_percent) + " prcnt are unique") 
    return valid_percent, valid_unique_percent

test_checkpoints()