# Image generation models
Implementation of different computer vision models, first on MNIST dataset (1x28x28) and then on higher dimensional images dataset such as AFHQ (animal faces high quality, 3x224x244).

Running the corresponding command trains the model and logs weights, losses, sampled images using tensorboard-pytorch.
`python run.py --config filepath_parameters_config`

## ResNet
Simple classification model

# GAN
Generative adverserial networks, a discriminator given an image predicts whether it comes from the training dataset or it was generated. The other model is a generator which randomly generate images, it is trained to dupe the discriminator. After training, we can use the generator to generate images.

## VAE
Encode image to two values, mu and logvar corresponding to a mean and a log-variance. These parameters are then used to sample from a Gaussian giving a vector which is decoded to get same dimension as input. Hence, after training, it is possible to generate using a normal Gaussian distribution and then decode this sample.

Different improvements can be implemented such as using quantized vectors instead of a Gaussian distribution and mulitnomial distribution to sample the quantized vectors which are then decoded as before.

## Diffusion models
Auto-regressive injection of noise to the images input until we get an encoding similar to the sampling of a gaussian normal and then decode it. Again we can generate new images at the gaussian distribution stage.