import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import Flatten, Reshape, InputLayer

class BetaVAE(Model):
    def __init__(self, latent_dim: int):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = 100

        self.inference_net = tf.keras.Sequential([
                InputLayer(input_shape=[28, 28, 1]),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                Flatten(),
                Dense(256, activation='relu'),
                # No activation
                Dense(self.latent_dim * 2),  # [means, stds]
            ])

        self.generative_net = tf.keras.Sequential([
                InputLayer(input_shape=[self.latent_dim]),
                Dense(256, activation='relu'),
                Dense(7 * 7 * 32, activation='relu'),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ])

    def encode(self, x):
        mean_logvar = self.inference_net(x)
        N = mean_logvar.shape[0]
        mean = tf.slice(mean_logvar, [0, 0], [N, self.latent_dim])
        logvar = tf.slice(mean_logvar, [0, self.latent_dim], [N, self.latent_dim])
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
