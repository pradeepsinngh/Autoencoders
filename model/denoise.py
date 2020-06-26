import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape

class DAE(tf.keras.Model):
    def __init__(self, latent_dim: int):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential([
                InputLayer(input_shape=[28, 28, 1]),
                Conv2D(filters=16, kernel_size=5, strides=(2, 2), activation='relu', padding="same"),
                Conv2D(filters=16, kernel_size=5, strides=(2, 2), activation='relu', padding="same"),
                Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding="same"),
                Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding="same"),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                # No activation
                Dense(self.latent_dim)
            ])

        self.generative_net = tf.keras.Sequential([
                InputLayer(input_shape=[self.latent_dim]),
                Dense(32, activation='relu'),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Reshape(target_shape=(2, 2, 16)),
                Conv2DTranspose(filters=16, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                Conv2DTranspose(filters=16, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                Conv2DTranspose(filters=16, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
                Conv2DTranspose(filters=16, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
                Conv2D(filters=1, kernel_size=1)
            ])

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        logits = self.generative_net(z)
        return logits
