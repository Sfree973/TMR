
from __future__ import division
import collections
import numpy as np
import sys
import time
import math
from Bi-BFS.Bi-BFS import BFS
from Bi-BFS.KB import KB
from itertools import count
from networks import discriminator_nn
import torch
import torch.nn as nn
import torch.optim as optim


LAMBDA = 5  # Gradient penalty lambda hyper-parameter.

class Discriminator(nn.Module):
    def __init__(self, batch_size, embedding_dim, learning_rate=0.001):
        super(Discriminator, self).__init__()

        self.initializer = nn.init.xavier_uniform_
        self.task = 'train'  # Placeholder for task (string)

        self.real_inputs = nn.Parameter(torch.Tensor(batch_size, 1, embedding_dim))
        self.fake_inputs = nn.Parameter(torch.Tensor(batch_size, 1, embedding_dim))

        self.disc_nn_real = discriminator_nn(embedding_dim, self.initializer)
        self.disc_nn_fake = discriminator_nn(embedding_dim, self.initializer)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.disc_nn_real(x)

    def predict(self, real, fake):
        self.eval()
        with torch.no_grad():
            disc_real = self.forward(real)
            disc_fake = self.forward(fake)
            original_disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)

            # WGAN Lipschitz penalty
            alpha = torch.rand(real.size(0), 1, 1)
            if torch.cuda.is_available():
                alpha = alpha.cuda()
            interpolate = alpha * self.real_inputs + (1 - alpha) * self.fake_inputs
            interpolate.requires_grad_(True)

            disc_interpolate = self.forward(interpolate)
            gradients = torch.autograd.grad(outputs=disc_interpolate, inputs=interpolate,
                                            grad_outputs=torch.ones(disc_interpolate.size()),
                                            create_graph=True, retain_graph=True)[0]

            slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
            gradient_penalty = torch.mean((slopes - 1.) ** 2)

            disc_cost = original_disc_cost + LAMBDA * gradient_penalty
            gen_reward = torch.mean(disc_fake)

            return disc_cost, gen_reward

    def update(self, real, fake):
        self.train()
        self.optimizer.zero_grad()
        disc_cost, _ = self.predict(real, fake)
        disc_cost.backward()
        self.optimizer.step()
        return disc_cost.item()

def discriminator_nn(embedding_dim, initializer):
    # Define your discriminator neural network architecture here
    # Note: You may need to modify the architecture to match your specific requirements
    # For simplicity, I'm assuming a simple linear layer followed by a sigmoid activation
    model = nn.Sequential(
        nn.Linear(embedding_dim, 1),
        nn.Sigmoid()
    )
    model.apply(initializer)
    return model
