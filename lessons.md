# Datasets
## CelebA
Downloaded from Kaggle, make sure that attributes file are in .txt and not .csv. Direct download from torchvision.datasets wasn't working, perhaps due to a download quota.

# ResNet
Strange behavior of model.eval(), mess up the batchnorm2d layers which leads to bad predictions.
# VAE Models




## TO INVESTIGATE
Cyclical Anneal Schedule for the beta, i.e. constant multiplying the Kuhlback-Leibler loss, c.f. https://github.com/larngroup/KL_divergence_loss


Weight for the KL-divergence loss important, good weight kl_weight = 0.0000025.

Furthermore, KL loss should be averaged over the batch, hence sum over all dimensions except first one, i.e. B.

Previously, unable to converge, outputs were simply grey (averaged colors), added convolutional residual blocks.

Requires lots of epochs, lr = 1e-4 quite small but it learns.


# MultiStage VAE
# Important Note
We need to detach the coarse output after the first stage before passing it through the second stage decoder, thus when optimizing the l1 loss will only update the weights of the second stage decoder.


### MNIST
Firstly, training with low kl weight (0.000025) and then retrain but with 100 times higher kl weight (0.0025).

Starting with too high kl weight leads to low variation and thus output of first stage ressembles an averaged input image.

First, 50 epochs with lr = 1e-4 and kl_weight = 0.00025, then 10 epochs with kl_weight = 0.0025


# GAN
Make sure when compute images sizes after convolutions or transposed convolutions to perform the modulo 2 before multiplying by 2.