from datetime import datetime

import torch
from torch.utils.data import Dataset
import pandas as pd
import os.path as osp
import numpy as np
from tqdm import tqdm
import spacy
import time

from models import DistMult, DistMultRegression


def negative_sampling(edge_index, num_nodes, eta=1):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(0) * eta) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_index.clone().repeat(eta, 1)
    neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
    neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)

    return neg_edge_index


class LiteralLinkPredDataset(Dataset):

    def __getitem__(self, index):
        # placeholder
        return None

    def __init__(self, triple_file, literal_file, transform=None, target_transform=None):
        print('start loading dataframes')
        df_triples_train = pd.read_csv(osp.join(triple_file, 'train.txt'), header=None, sep='\t')
        df_triples_val = pd.read_csv(osp.join(triple_file, 'valid.txt'), header=None, sep='\t')
        df_triples_test = pd.read_csv(osp.join(triple_file, 'test.txt'), header=None, sep='\t')
        df_literals_num = pd.read_csv(osp.join(triple_file, 'numerical_literals.txt'), header=None, sep='\t')
        df_literals_txt = pd.read_csv(osp.join(triple_file, 'text_literals.txt'), header=None, sep='\t')

        print('start loading relational data')
        self.entities = list(set(np.concatenate([df_triples_train[0].unique(),
                                                 df_triples_test[0].unique(),
                                                 df_triples_val[0].unique(),
                                                 df_triples_train[2].unique(),
                                                 df_triples_test[2].unique(),
                                                 df_triples_val[2].unique(),
                                                 df_literals_num[0].unique(),
                                                 df_literals_txt[0].unique()])))
        self.num_entities = len(self.entities)

        self.relations = list(set(np.concatenate([df_triples_train[1].unique(),
                                                  df_triples_test[1].unique(),
                                                  df_triples_val[1].unique()])))
        self.num_relations = len(self.relations)

        self.entity2id = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation2id = {self.relations[i]: i for i in range(len(self.relations))}

        self.edge_index_train = torch.stack([torch.tensor(df_triples_train[0].map(self.entity2id)),
                                             torch.tensor(df_triples_train[2].map(self.entity2id))])
        self.edge_index_val = torch.stack([torch.tensor(df_triples_val[0].map(self.entity2id)),
                                           torch.tensor(df_triples_val[2].map(self.entity2id))])
        self.edge_index_test = torch.stack([torch.tensor(df_triples_test[0].map(self.entity2id)),
                                            torch.tensor(df_triples_test[2].map(self.entity2id))])

        self.edge_type_train = torch.tensor(df_triples_train[1].map(self.relation2id))
        self.edge_type_val = torch.tensor(df_triples_val[1].map(self.relation2id))
        self.edge_type_test = torch.tensor(df_triples_test[1].map(self.relation2id))

        # with E = number of embeddings, R = number of attributive relations, V = feature dim
        print('start loading numerical literals: E x R')
        self.attr_relations_num = list(df_literals_num[1].unique())
        self.attr_relation_2_id_num = {self.attr_relations_num[i]: i for i in range(len(self.attr_relations_num))}

        df_literals_num[0] = df_literals_num[0].map(self.entity2id).astype(int)
        df_literals_num[1] = df_literals_num[1].map(self.attr_relation_2_id_num).astype(int)
        df_literals_num[2] = df_literals_num[2].astype(float)

        self.num_attributive_relations_num = len(self.attr_relations_num)

        self.features_num = []
        for i in tqdm(range(len(self.entities))):
            df_i = df_literals_num[df_literals_num[0] == i]

            feature_i = torch.zeros(self.num_attributive_relations_num)
            for index, row in df_i.iterrows():
                feature_i[int(row[1])] = float(row[2])

            self.features_num.append(feature_i)
        self.features_num = torch.stack(self.features_num)

        max_lit, min_lit = torch.max(self.features_num, dim=0).values, torch.min(self.features_num, dim=0).values

        self.features_num = (self.features_num - min_lit) / (max_lit - min_lit + 1e-8)

        print('start loading textual literals: E x R x V')
        self.attr_relations_txt = list(df_literals_txt[1].unique())
        self.attr_relation_2_id_txt = {self.attr_relations_txt[i]: i for i in range(len(self.attr_relations_txt))}

        df_literals_txt[0] = df_literals_txt[0].map(self.entity2id).astype(int)
        df_literals_txt[1] = df_literals_txt[1].map(self.attr_relation_2_id_txt).astype(int)
        df_literals_num[2] = df_literals_num[2].astype(str)
        self.num_attributive_relations_txt = len(self.attr_relations_txt)
        nlp = spacy.load('en_core_web_md')

        self.features_txt = []
        for i in tqdm(range(len(self.entities))):
            df_i = df_literals_txt[df_literals_txt[0] == i]

            features_txt_i = torch.zeros(self.num_attributive_relations_txt, 300)
            for index, row in df_i.iterrows():
                spacy_sentence = torch.tensor(nlp(row[2]).vector)
                features_txt_i[int(row[1])] = spacy_sentence

            self.features_txt.append(features_txt_i)

        self.features_txt = torch.stack(self.features_txt)


def train_standard_lp(eta=2, regularization=False, literal_features_alpha=0.05):
    model.train()
    start = time.time()

    edge_index_batches = torch.split(train_edge_index_t, 1000)
    edge_type_batches = torch.split(train_edge_type, 1000)

    indices = np.arange(len(edge_index_batches))
    np.random.shuffle(indices)

    loss_total = 0
    for i in indices:
        edge_idxs, relation_idx = edge_index_batches[i], edge_type_batches[i]
        optimizer.zero_grad()

        edge_idxs_neg = negative_sampling(edge_idxs, dataset.num_entities, eta=eta)

        out_pos = model.forward(edge_idxs[:, 0], relation_idx, edge_idxs[:, 1])
        out_neg = model.forward(edge_idxs_neg[:, 0], relation_idx.repeat(eta), edge_idxs_neg[:, 1])

        out = torch.cat([out_pos, out_neg], dim=0)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * eta)], dim=0).to(DEVICE)

        loss = loss_function_model(out, gt)

        if regularization:
               loss += 0.000001 * model.l3_regularization()

        if literal_features_alpha > 0:
            batch_entities = torch.tensor(list(set(edge_idxs[:, 0].tolist() + edge_idxs[:, 1].tolist()))).to(DEVICE)
            out = model_features.forward(model.entity(batch_entities))
            feature_loss = loss_function_features(out, features_num[batch_entities])
            loss = loss + literal_features_alpha * feature_loss

        loss_total += loss
        loss.backward()
        optimizer.step()
    end = time.time()
    print('esapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))


@torch.no_grad()
def compute_rank(ranks):
    # print(ranks)
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr_triple_scoring(model, eval_edge_index, eval_edge_type,
                               fast=False):
    model.eval()
    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for triple_index in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(dataset.num_entities)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        out = model.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(dataset.num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (dataset.edge_index_train, dataset.edge_type_train),
            (dataset.edge_index_val, dataset.edge_type_val),
            (dataset.edge_index_test, dataset.edge_type_test),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(dataset.num_entities)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        out = model.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), \
           ranks.mean(), \
           ranks[ranks <= 10].size(0) / num_ranks, \
           ranks[ranks <= 5].size(0) / num_ranks, \
           ranks[ranks <= 3].size(0) / num_ranks, \
           ranks[ranks <= 1].size(0) / num_ranks


if __name__ == '__main__':
    if not osp.isfile('./data/fb15k-237/processed.pt'):
        dataset = LiteralLinkPredDataset('./data/fb15k-237', './literals.txt')
        torch.save(dataset, './data/fb15k-237/processed.pt')
    else:
        dataset = torch.load('./data/fb15k-237/processed.pt')

    EMBEDDING_DIM = 200
    DEVICE = torch.device('cuda')
    lr = 0.0005
    BATCH_SIZE = 128
    run_name = 'feature_vs_objective_link_pred_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = DistMult(dataset.num_entities, dataset.num_relations, EMBEDDING_DIM)
    model.to(DEVICE)
    model_features = DistMultRegression(EMBEDDING_DIM, dataset.num_attributive_relations_num)
    model_features.to(DEVICE)

    loss_function_model = torch.nn.BCELoss()  # torch.nn.MSELoss()
    loss_function_features = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(list(model.parameters()) + list(model_features.parameters()), lr=0.0005)
    model.train()
    model_features.train()

    train_edge_index_t = dataset.edge_index_train.t().to(DEVICE)
    train_edge_type = dataset.edge_type_train.to(DEVICE)
    features_num = dataset.features_num.to(DEVICE)

    for epoch in range(0, 1000):
        train_standard_lp(eta=1, literal_features_alpha=0)
        if epoch % 50 == 0:
            mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                              dataset.edge_index_val,
                                                                              dataset.edge_type_val,
                                                                              fast=True)
            print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                      dataset.edge_index_test,
                                                                      dataset.edge_type_test,
                                                                      fast=True)
    print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)



