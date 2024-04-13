# VAE Models
Weight for the KL-divergence loss important, good weight kl_weight = 0.0000025.

Furthermore, KL loss should be averaged over the batch, hence sum over all dimensions except first one, i.e. B.

Previously, unable to converge, outputs were simply grey (averaged colors), added convolutional residual blocks.

Requires lots of epochs, lr = 1e-4 quite small but it learns.
