# EP42 — Generative Adversarial Networks (GANs)

> **Deep Learning Series · Episode 42 of 72 · Module 8 — Generative Deep Learning**

A completely different generation strategy from the VAE:  
two networks locked in a game.  
The Generator tries to fool the Discriminator; the Discriminator tries to catch every fake.  
Neither is ever told what a good image looks like.

## What you will learn

- The adversarial game (G vs D)
- Generator architecture (ConvTranspose2d, BatchNorm, ReLU, Tanh)
- Discriminator architecture (Conv2d, BatchNorm, LeakyReLU, Sigmoid)
- Minimax vs non-saturating loss
- Why GAN training is unstable (vanishing gradients, oscillation)
- Mode collapse — how to spot it
- DCGAN stabilisation recipe
- Wasserstein GAN (overview)
- Full PyTorch DCGAN on MNIST

## Key formulas
