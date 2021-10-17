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

from image_creator import valid_room

BATCH_SIZE = 16
IMAGE_SIZE = (7, 7)
NUMBER_OF_TILE_TYPES = 4
DATASET_PATH = "progen_images/numpy_images.npy"

# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#    directory=DATASET_PATH, color_mode='grayscale', batch_size=BATCH_SIZE, image_size=(IMAGE_SIZE), label_mode=None, interpolation='nearest'
#)

def transform_from_one_hot(one_hot_image):
  normal_image = np.argmax(one_hot_image, axis=-1)
  normal_image = np.expand_dims(normal_image, -1)
  return normal_image

def load_dataset_as_one_hot(path):
    images = np.load(path)

    one_hot_shape = list(images.shape)
    one_hot_shape[-1] = NUMBER_OF_TILE_TYPES
    one_hot_images = np.zeros(shape=one_hot_shape)

    for i in range(images.shape[0]):
      for j in range(images.shape[1]):
        for k in range(images.shape[2]):
          index = images[i, j, k, 0]
          one_hot_images[i, j, k, index] = 1

    # needed because dataset_from_slices reduce first dim
    return np.expand_dims(one_hot_images, 1).astype(np.float32)


numpy_dataset = load_dataset_as_one_hot(DATASET_PATH)

a = transform_from_one_hot(numpy_dataset[0])

data_tensor = tf.convert_to_tensor(numpy_dataset, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices(data_tensor)

#dataset = dataset.shuffle(1000)#.batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1*1*16, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((1, 1, 16)))
    assert model.output_shape == (None, 1, 1, 16)

    #model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Conv2DTranspose(32, (5, 5)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same'))
    #model.add(layers.Softmax())
    #model.add(layers.Softmax())

    assert model.output_shape == (None, 7, 7, NUMBER_OF_TILE_TYPES)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same', input_shape=[7, 7, NUMBER_OF_TILE_TYPES]))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (5, 5)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

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

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-6)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 1000000
noise_dim = 100
num_examples_to_generate = 10

valid = 0

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, update_discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
  
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      #numpy_output = generated_images.numpy()
      #print("Average generation value: ", np.average(numpy_output))

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    if update_discriminator:
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return (gen_loss, disc_loss)

def generate_boards():
  for _ in range (num_examples_to_generate):
    generation_noise = np.random.rand(1, noise_dim).astype(np.float32)
    generation_noise = tf.convert_to_tensor(generation_noise, tf.float32)
    output = generator(generation_noise)

    #print(generation_noise)

    numpy_output = output.numpy()

    #print("Average generation value: ", np.average(numpy_output))

    global valid

    for i in range(numpy_output.shape[0]):
      room = np.expand_dims(numpy_output[0], 0)
      room = transform_from_one_hot(room)
      #print(transform_from_one_hot(numpy_output[0]))
      if(valid_room(room)):
        np.save("generated_images/validroom" + str(valid), room)
        valid += 1

DISCRIMINATOR_UPDATE_EPOCHS = 8

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss, desc_loss = train_step(image_batch, epoch % DISCRIMINATOR_UPDATE_EPOCHS == 0)

    # Save the model every 1000 epochs and try to generate valid boards
    if (epoch + 1) % 200 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      generate_boards()
      print ("gen_loss:", gen_loss.numpy(), "desc_loss", desc_loss.numpy(), 'Epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(dataset, EPOCHS)
