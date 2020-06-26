# parser_args code referred the hwalseoklee's code:
# https://github.com/hwalsuklee/tensorflow-data-VAE/blob/master/run_main.py

import tensorflow as tf
from utils import data, plot
from model.auto_encoder import AE
from model.variational_autoenc import VAE, CVAE
from model.BetaVAE import BetaVAE
from loss import compute_loss
import time
import argparse

def parse_args():
    desc = "Tensorflow 2.0 implementation of 'AutoEncoder Families (AE, VAE, CVAE(Conditional VAE))'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ae_type', type=str, default=False,
                        help='Type of autoencoder: [AE, DAE, VAE, CVAE, BetaVAE]')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Degree of latent dimension(a.k.a. "z")')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='The number of training epochs')
    parser.add_argument('--learn_rate', type=float, default=1e-4,
                        help='Learning rate during training')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    return parser.parse_args()


def train(ae_type, latent_dim=2, epochs=100, lr=1e-4, batch_size=1000):

    if ae_type == "AE" or ae_type == "DAE":
        model = AE(latent_dim)
    elif ae_type == "VAE":
        model = VAE(latent_dim)
    elif ae_type == "CVAE":
        model = CVAE(latent_dim)
    elif ae_type == "BetaVAE":
        model = BetaVAE(latent_dim)
    else:
        raise ValueError

    # load train and test data
    train_dataset, test_dataset = data.load_dataset(ae_type, batch_size=batch_size)
    # initialize Adam optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(1, epochs + 1):
        last_loss = 0

        for train_x, train_y in train_dataset:
            gradients, loss = compute_gradients(model, train_x, train_y, ae_type)
            apply_gradients(optimizer, gradients, model.trainable_variables)
            last_loss = loss

        if epoch % 2 == 0:
            print('Epoch {}, Loss: {}'.format(epoch, last_loss))

    return model


def compute_gradients(model, x, y, ae_type):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y, ae_type)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def main(args):

    train(latent_dim=args.latent_dim, epochs=args.num_epochs, lr=args.learn_rate,
            batch_size=args.batch_size, ae_type = args.ae_type)

if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)
