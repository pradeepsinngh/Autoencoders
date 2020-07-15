# Autoencoders in TensorFlow 2.0
Implementations of following Autoencoders;
- Vanilla Autoencoder (AE)
- Denoise Autoencoder
- Sparese Autoencoder (in progress)
- Contractive Autoencoder (in progress)
- Variational Autoencoder (VAE)
- Conditional Variational Autoencoder (CVAE)
- Beat Variational Autoencoder (beta-VAE) (in progress)

## How to run:

run ```python3 main.py --ae_type AE```

### Parameters that we can pass: 
 - ae_type: Type of autoencoder - AE, DAE, VAE, CVAE, BetaVAE
 - latent_dim: Degree of latent dimension - 2, 3, etc.
 - num_epochs: The number of training epochs - 100 etc.
 - learn_rate: Learning rate during training - 1e-4
 - batch_size: Batch size - 1000
