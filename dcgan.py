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

import tensorflow as tf
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

from IPython import display

BUFFER_SIZE = 60000
BATCH_SIZE = 8
IMAGE_SIZE = (7, 7)
NUMBER_OF_TILE_TYPES = 6
DATASET_PATH = "progen_images/numpy_images.npy"

# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#    directory=DATASET_PATH, color_mode='grayscale', batch_size=BATCH_SIZE, image_size=(IMAGE_SIZE), label_mode=None, interpolation='nearest'
#)

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

data_tensor = tf.convert_to_tensor(numpy_dataset, dtype=tf.float32)
print(data_tensor)
dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
#dataset = tf.data.Dataset(data_tensor)

#print(dataset)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*16, use_bias=False, input_shape=(100,)))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((5, 5, 16)))
    assert model.output_shape == (None, 5, 5, 16)

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(32, (3, 3)))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (5, 5), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same'))
    model.add(layers.Softmax())

    assert model.output_shape == (None, 7, 7, NUMBER_OF_TILE_TYPES)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NUMBER_OF_TILE_TYPES, (3, 3), padding='same', input_shape=[7, 7, NUMBER_OF_TILE_TYPES]))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(32, (5, 5), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(64, (5, 5)))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(64))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dense(1)
    )
    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = make_discriminator_model()

np_output = generated_image.numpy() 

decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 30
noise_dim = 100
num_examples_to_generate = 4

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Save the model every 1000 epochs and try to generate valid boards
    if (epoch + 1) % 1000 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
#def display_image(epoch_no):
#  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
#
#display_image(EPOCHS)
#
#anim_file = 'dcgan.gif'
#
#with imageio.get_writer(anim_file, mode='I') as writer:
#  filenames = glob.glob('image*.png')
#  filenames = sorted(filenames)
#  for filename in filenames:
#    image = imageio.imread(filename)
#    writer.append_data(image)
#  image = imageio.imread(filename)
#  writer.append_data(image)

#import tensorflow_docs.vis.embed as embed
#embed.embed_file(anim_file)
