from keras import layers
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
T = 1300
import math
batch_size = 60
image_size = 32
DEFAULT_DTYPE = tf.float32
current_time = []

####
#These references helped with the implmentation of the Unet architecture.
#https://keras.io/examples/generative/ddim/
#https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac
#https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/scripts/run_celebahq.py#L43
####

# this function will help us set the RNG key for Numpy
def set_key(key):
    np.random.seed(key)


#We will use a linear beta schedule.
def linear_beta_schedule(timesteps = 1000, start=0.0001,end = 0.02):
    return np.linspace(start,end,timesteps)

def generate_timestamp(key,num):
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=T, dtype=tf.int32)

beta = np.linspace(0.0001, 0.02, T)

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)

def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = tf.exp(tf.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
  # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
    emb = tf.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == [len(timesteps), embedding_dim]
  return emb

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        current_time.append(1)
        self.network = get_network(image_size)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def forwardDiffusion(self,x_0, t, tsrng):
        set_key(tsrng)
        noise = tf.random.normal((batch_size,image_size, image_size,1))
        sqrt_alpha_cumprod_reshape = tf.cast(tf.reshape(tf.gather(alpha_bar, t), (-1, 1, 1, 1)), dtype ='float32')
        one_minus_sqrt_alpha_cumprod_reshape = tf.cast(tf.reshape(tf.gather(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)), dtype='float32')
        image_with_noise = sqrt_alpha_cumprod_reshape * x_0 + one_minus_sqrt_alpha_cumprod_reshape * noise
        assert x_0.shape == image_with_noise.shape
        return image_with_noise, noise

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        rng, tsrng = np.random.randint(0, 100000, size=(2,))
        # sample uniform random diffusion times
        diffusion_times = generate_timestamp(rng, batch_size)

        images_with_noise, noise = self.forwardDiffusion(images, diffusion_times, tsrng)

        current_time.clear()
        current_time.append(diffusion_times)

        with tf.GradientTape() as tape:
            pred_noises = self.network(images_with_noise, training=True)
            loss = self.loss(noise, pred_noises)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics[:]}

    def ddim(self, x_t, pred_noise, t, sigma_t):
        alpha_t_bar = np.take(alpha_bar, t)
        alpha_t_minus_one = np.take(alpha, t - 1)

        pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise) / (alpha_t_bar ** 0.5)
        pred = (alpha_t_minus_one ** 0.5) * pred

        pred = pred + ((1 - alpha_t_minus_one - (sigma_t ** 2)) ** 0.5) * pred_noise
        eps_t = np.random.normal(size=x_t.shape)
        pred = pred + (sigma_t * eps_t)

        return pred

    def sampling_ddim(self):

        # Define number of inference loops to run
        inference_timesteps = 10

        # Create a range of inference steps that the output should be sampled at
        inference_range = range(0, T, T // inference_timesteps)

        x = tf.random.normal((1, image_size, image_size, 1))
        img_list = []
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))

        # Iterate over inference_timesteps
        for index, i in enumerate(reversed(range(inference_timesteps))):
            t = np.expand_dims(inference_range[i], 0)

            pred_noise = self.network(x, training =False)

            current_time.clear()
            current_time.append(t)
            x = self.ddim(x, pred_noise, t, 0)

        plt.imshow(x[0,:,:]+1.0*127.5, cmap="gray")
        plt.show()


    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        self.sampling_ddim()
        plt.close()



# Activation Funciton swift swift(x) = x * sigmoid(x)
def swift(x):
    return layers.Activation(tf.nn.swish)(x)


# Redidual Block described in the report
def ResidualBlock(width, temb):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x += layers.Dense(units=width)(swift(temb))
        x = layers.Add()([x, residual])
        return x

    return apply

#Encoder
def DownBlock(width, block_depth, temb):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, temb)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

#Decoder
def UpBlock(width, block_depth, temb):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, temb)(x)
        return x

    return apply

#Defind our unet network
def get_network(image_size, widths = [32, 64, 128, 256], block_depth =2):
    noisy_images = keras.Input(shape=(image_size, image_size, 1))

    #Postinal time embeddings
    temb = get_timestep_embedding(current_time, image_size)
    temb = layers.Dense(units=image_size * 4)(temb)
    temb = layers.Dense(units=image_size * 4)(swift(temb))

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, temb = temb)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], temb = temb)(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, temb = temb)([x, skips])

    x = layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(noisy_images, x, name="residual_unet")