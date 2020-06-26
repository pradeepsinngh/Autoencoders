import tensorflow as tf

def compute_loss(model, x, y, ae_type):

    if ae_type == "AE":
        return binaryCrossEntropy(model, x, y)
    elif ae_type == "VAE" or ae_type == "CVAE" or ae_type == "BetaVAE":
        return VAELoss(model, x, y, ae_type)
    else:
        raise ValueError

def binaryCrossEntropy(model, x, y):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    z = model.encode(x)
    x_logits = model.decode(z)
    loss = loss_object(x, x_logits)
    return loss

def VAELoss(model, x, y, ae_type):

    if ae_type == "VAE":
        mean, logvar = model.encode(x)
    elif ae_type == "CVAE":
        mean, logvar = model.encode(x, y)
    elif ae_type == "BetaVAE":
        mean, logvar = model.encode(x)
    else:
        raise ValueError

    z = trick(mean, logvar)

    if ae_type == "VAE":
        x_logits  = model.decode(z)
    elif ae_type == "CVAE":
        x_logits  = model.decode(z, y)
    elif ae_type == "BetaVAE":
        x_logits = model.decode(z)
    else:
        raise ValueError

    # cross_ent = - marginal likelihood
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
    marginal_likelihood = - tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

    KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
    KL_divergence = tf.reduce_mean(KL_divergence)

    if ae_type == "BetaVAE":
        ELBO = marginal_likelihood - model.beta * KL_divergence
    else:
        ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO
    return loss

# Reparametrization trick
def trick(mean, logvar):
    eps = tf.random.normal(shape = mean.shape)
    return eps * tf.exp(logvar * .5) + mean
