import tensorflow as tf
from utils import reparameterize

def compute_loss(model, x, ae_type):
    if ae_type == "AE":
        return binaryCrossEntropy(model, x)
    elif ae_type == "VAE" or ae_type == "CVAE":
        return VAELoss(model, x)
    else:
        raise ValueError

def binaryCrossEntropy(model, x):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    z = model.encode(x)
    x_logits = model.decode(z)
    loss = loss_object(x, x_logits)
    return loss

def VAELoss(model, x):
    mean, logvar = model.encode(x)
    z = reparameterize.trick(mean, logvar)
    x_logits = model.decode(z)

    # cross_ent = - marginal likelihood
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
    marginal_likelihood = - tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

    KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO
    return loss
