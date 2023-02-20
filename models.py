import torch
from torch import nn
from torch.nn import Parameter


class DistMult (nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel = torch.nn.Embedding(num_relations, embedding_dim)

    def forward(self, head_idx, rel_idx, tail_idx):

        h_head = self.entity(head_idx)
        h_relation = self.rel(rel_idx)
        h_tail = self.entity(tail_idx)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out

    def l3_regularization(self):
        return (self.head.weight.norm(p=3) ** 3 + self.rel.weight.norm(p=3) ** 3)


class DistMultRegression (nn.Module):
    def __init__(self, embedding_dim, num_attributive_relations):
        super().__init__()
        self.W = torch.nn.Linear(embedding_dim, num_attributive_relations)

    def forward(self, entity_embedding):
        return self.W(entity_embedding)
