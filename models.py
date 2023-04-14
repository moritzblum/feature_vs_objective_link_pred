import torch
from torch import nn
from torch.autograd import Variable
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
        return torch.sigmoid(self.W(entity_embedding))



class DistMultLiteral_gate(torch.nn.Module):

    def __init__(self, num_entities, num_relations, embedding_dim, numerical_literals, dropout=0.2, batch_norm=False):
        super(DistMultLiteral_gate, self).__init__()

        self.emb_dim = embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = Gate(self.emb_dim + self.n_num_lit, self.emb_dim)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(self.emb_e.weight, self.numerical_literals)

        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb * rel_emb, e2_multi_emb.t())
        pred = torch.sigmoid(pred)

        return pred


class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 # gate_activation=nn.functional.softmax):
                 gate_activation=nn.functional.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output