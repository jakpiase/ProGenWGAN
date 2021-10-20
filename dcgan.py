#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import ge
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import sigmoid
from tensorflow.python.keras.layers.core import Activation
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from board_operations import *

from image_creator import valid_room

BATCH_SIZE = 64
IMAGE_SIZE = (7, 7)
NUMBER_OF_TILE_TYPES = 7
DATASET_PATH = "all_datasets/numpy_images.npy"




#TESTING SHUFFLE AND STUFF
#BUFFER_SIZE = 1000
#dataset = dataset.shuffle(4 * BATCH_SIZE)


#dataset = dataset.shuffle(1000)#.batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1*1*16, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((1, 1, 16)))
    assert model.output_shape == (None, 1, 1, 16)

    model.add(layers.Conv2DTranspose(64, (5, 5)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same'))
    model.add(layers.Softmax())
    #model.add(layers.Softmax())

    assert model.output_shape == (None, 7, 7, NUMBER_OF_TILE_TYPES)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same', input_shape=[7, 7, NUMBER_OF_TILE_TYPES]))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (5, 5)))
    #model.add(layers.Activation(activation="sigmoid"))   
    #model.add(layers.LeakyReLU(alpha=0.2))

#    model.add(layers.Flatten())
#
#    model.add(layers.Dense(32))
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU(alpha=0.2))
#    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

generator.summary()
discriminator.summary()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

counter = 0
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(2e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 10000
noise_dim = 100
num_examples_to_generate = 100

valid = 0

random_generator = tf.random.Generator.from_non_deterministic_state()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, update_discriminator):
    noise = random_generator.normal([BATCH_SIZE, noise_dim])
  
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      #numpy_output = generated_images.numpy()
      #print("Average generation value: ", np.average(numpy_output))

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    if update_discriminator:
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return (gen_loss, disc_loss)


numpy_base_dataset = np.load(DATASET_PATH)

#print(numpy_base_dataset)
#print(numpy_base_dataset.shape)
#for i in range(numpy_base_dataset.shape[0]):
#  print(np.reshape(numpy_base_dataset[i], (7, 7)))
#
#print(numpy_base_dataset.shape)

def generate_boards():
  generation_noise = random_generator.normal([num_examples_to_generate, noise_dim])
  output = generator(generation_noise)

  numpy_output = output.numpy()

    #print("Average generation value: ", np.average(numpy_output))

  global valid

  room_list = []
  valid_rooms = 0

  for i in range(numpy_output.shape[0]):
    room = np.expand_dims(numpy_output[i], 0)
    room = transform_from_one_hot(room)
    room_list.append(room)
    if valid_room(room):
      valid_rooms += 1
      if in_dataset(numpy_base_dataset, room) == False:
        np.save("generated_images/valid_unique_room" + str(valid), room)
        valid += 1

  return str(f"{len(get_unique(room_list)) / float(num_examples_to_generate):.0%}") + " valid rooms generated:" + str(valid_rooms) + " "

DISCRIMINATOR_UPDATE_EPOCHS = 1

def train(dataset, epochs):
  start = time.time()
  for epoch in range(epochs):

    for image_batch in dataset:
      gen_loss, desc_loss = train_step(image_batch, epoch % DISCRIMINATOR_UPDATE_EPOCHS == 0)

    # Save the model every X epochs and try to generate valid boards
    #if (epoch + 1) % 1:
    #  generate_boards()
    if (epoch + 1) % 1 == 0:
      print ("gen_loss:", gen_loss.numpy(), "desc_loss", desc_loss.numpy(), "unique rooms generated:", generate_boards(), 'Epoch {} took {} sec'.format(epoch + 1, time.time()-start))
      #generate_boards()
      checkpoint.save(file_prefix = checkpoint_prefix)
      start = time.time()

numpy_dataset = load_dataset_as_one_hot(DATASET_PATH)

print("Found", numpy_dataset.shape[0], "images in dataset")

data_tensor = tf.convert_to_tensor(numpy_dataset, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices(data_tensor)

train(dataset, EPOCHS)
