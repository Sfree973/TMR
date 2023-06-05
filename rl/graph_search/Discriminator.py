
from __future__ import division

import collections
import numpy as np
import sys
import time
import math
from itertools import count
from networks import discriminator_nn
import torch
import torch.nn as nn
import torch.optim as optim
from Sampling import sampling  


LAMBDA = 5  # Gradient penalty lambda hyper-parameter.



class Discriminator(nn.Module):
    def __init__(self, batch_size, embedding_dim, learning_rate=0.001):
        super(Discriminator, self).__init__()

        self.initializer = nn.init.xavier_uniform_
        self.task = ''
        
        self.real_inputs = torch.zeros(batch_size, 1, embedding_dim)
        self.fake_inputs = torch.zeros(batch_size, 1, embedding_dim)
        
        self.disc_real = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.disc_fake = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
    
    def forward(self, inputs):
        disc_real_out = self.disc_real(inputs)
        disc_fake_out = self.disc_fake(inputs)
        return disc_real_out, disc_fake_out
    
    def predict(self, real, fake):
        disc_real_out, disc_fake_out = self.forward(real)
        original_disc_cost = torch.mean(disc_fake_out) - torch.mean(disc_real_out)
        
        alpha = torch.rand(real.shape[0], 1, 1)
        differences = fake - real
        interpolate = real + (alpha * differences)

        interpolate_out = self.forward(interpolate)[0]
        gradients = torch.autograd.grad(outputs=interpolate_out, inputs=interpolate,
                                        grad_outputs=torch.ones(interpolate_out.size()),
                                        create_graph=True, retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
        gradient_penalty = torch.mean((slopes - 1.) ** 2)

        disc_cost = original_disc_cost + LAMBDA * gradient_penalty
        gen_reward = torch.mean(disc_fake_out)

        return disc_cost, gen_reward
    
    def update(self, real, fake):
        self.optimizer.zero_grad()
        loss, _ = self.predict(real, fake)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def set_task(self, task):
        self.task = task

    def set_real_inputs(self, real_inputs):
        self.real_inputs = real_inputs

    def set_fake_inputs(self, fake_inputs):
        self.fake_inputs = fake_inputs

    def train(self):
        self.train()

    def eval(self):
        self.eval()
