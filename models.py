import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_


class DistMult (nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.2, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.entity = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel = torch.nn.Embedding(num_relations, embedding_dim)

        self.dp_ent = torch.nn.Dropout(dropout)
        self.dp_rel = torch.nn.Dropout(dropout)

        self.bn_head = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_rel = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_tail = torch.nn.BatchNorm1d(embedding_dim)

    def init(self):
        xavier_normal_(self.entity.weight.data)
        xavier_normal_(self.rel.weight.data)

    def forward(self, head_idx, rel_idx, tail_idx):

        h_head = self.dp_ent(self.entity(head_idx))
        h_relation = self.dp_rel(self.rel(rel_idx))
        h_tail = self.dp_ent(self.entity(tail_idx))

        if self.batch_norm:
            h_head = self.bn_head(h_head)
            h_relation = self.bn_rel(h_relation)
            h_tail = self.bn_tail(h_tail)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out

    def l3_regularization(self):
        return (self.entity.weight.norm(p=3) ** 3 + self.rel.weight.norm(p=3) ** 3)


class DistMultRegression (nn.Module):
    def __init__(self, embedding_dim, num_attributive_relations):
        super().__init__()
        self.W = torch.nn.Linear(embedding_dim, num_attributive_relations)

    def forward(self, entity_embedding):
        return self.W(entity_embedding)
