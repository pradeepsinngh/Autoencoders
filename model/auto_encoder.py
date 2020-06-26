# Some of codes referred to tensorflow 2.0 official tutorial for VAE
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape

class AE(tf.keras.Model):
    def __init__(self, latent_dim: int):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential([
                InputLayer(input_shape=[28, 28, 1]),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                # No activation
                Dense(self.latent_dim),
            ])

        self.generative_net = tf.keras.Sequential([
                InputLayer(input_shape=[self.latent_dim]),
                Dense(64, activation='relu'),
                Dense(7 * 7 * 32, activation='relu'),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(filters=64,kernel_size=3,strides=(2, 2),padding="SAME",activation='relu'),
                Conv2DTranspose(filters=32,kernel_size=3,strides=(2, 2),padding="SAME",activation='relu'),
                # No activation
                Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid'),
            ])

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        logits = self.generative_net(z)
        return logits
