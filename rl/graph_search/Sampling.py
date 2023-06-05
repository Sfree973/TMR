
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
from torch.utils.data import Dataset, DataLoader

# Import functions from rule_learning.py and rule_application.py
from rule_learning import learn_rules
from rule_application import apply_rules, select_high_score_paths

class KGDataset(Dataset):
    def __init__(self, path):
        f = open(path)
        content = f.readlines()
        f.close()
        kb = KB()
        for line in content:
            ent1, rel, ent2 = line.rsplit()
            kb.addRelation(ent1, rel, ent2)
        self.kb = kb

        f = open(relationPath)
        train_data = f.readlines()
        f.close()

        self.samples = []
        for sample in train_data:
            ent1, rel, ent2 = sample.split()
            self.samples.append((ent1, rel, ent2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ent1, rel, ent2 = self.samples[idx]
        return ent1, rel, ent2

class RuleNet(nn.Module):
    def __init__(self, num_entities, num_relations, rule_lengths):
        super(RuleNet, self).__init__()
        self.rule_lengths = rule_lengths

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.rule_scores = nn.ModuleDict()
        for rule_length in rule_lengths:
            self.rule_scores[str(rule_length)] = nn.Linear(embedding_dim * rule_length, 1)

    def forward(self, ent1, rel, ent2, rule_walks):
        ent1_emb = self.entity_embeddings(ent1)
        rel_emb = self.relation_embeddings(rel)
        ent2_emb = self.entity_embeddings(ent2)

        rule_scores = []
        for rule_length in self.rule_lengths:
            rule_score = self.rule_scores[str(rule_length)](rule_walks[:, :rule_length].reshape(-1, embedding_dim * rule_length))
            rule_scores.append(rule_score)
        rule_scores = torch.cat(rule_scores, dim=1)

        scores = torch.sum(ent1_emb * rel_emb * ent2_emb, dim=1) + rule_scores.squeeze(1)
        return scores

def sampling(path_threshold=2, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)

    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    num_samples = len(train_data)
    demo_path_dict = {}
    for episode in range(num_samples):
        sample = train_data[episode % num_samples].split()
        ent1 = sample[0]
        ent2 = sample[2]
        rel = sample[1]

        kb.removePath(ent1, ent2)
        try:
            suc, entity_list, path_list = BFS(kb, ent1, ent2)
            path_str = ' -> '.join(path_list)
        except Exception as e:
            print('Episode %d' % episode)
            print('Cannot find a path')
            continue

        if path_str not in demo_path
