from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import spacy


class LiteralLinkPredDataset(Dataset):

    def __getitem__(self, index):
        # placeholder
        return None

    def __init__(self, triple_file, transform=None, target_transform=None):
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
        self.features_num_mask = []
        for i in tqdm(range(len(self.entities))):
            df_i = df_literals_num[df_literals_num[0] == i]

            feature_i = torch.zeros(self.num_attributive_relations_num)
            feature_i_mask = torch.zeros(self.num_attributive_relations_num, dtype=torch.bool)
            for index, row in df_i.iterrows():
                feature_i[int(row[1])] = float(row[2])
                feature_i_mask[int(row[1])] = True

            self.features_num.append(feature_i)
            self.features_num_mask.append(feature_i_mask)
        self.features_num = torch.stack(self.features_num)
        self.features_num_mask = torch.stack(self.features_num_mask)

        max_lit, min_lit = torch.max(self.features_num, dim=0).values, torch.min(self.features_num, dim=0).values

        self.features_num = (self.features_num - min_lit) / (max_lit - min_lit + 1e-8)


        print('start loading textual literals: E x R x V')
        #self.attr_relations_txt = list(df_literals_txt[1].unique())
        #self.attr_relation_2_id_txt = {self.attr_relations_txt[i]: i for i in range(len(self.attr_relations_txt))}

        #df_literals_txt[0] = df_literals_txt[0].map(self.entity2id).astype(int)
        #df_literals_txt[1] = df_literals_txt[1].map(self.attr_relation_2_id_txt).astype(int)
        #df_literals_num[2] = df_literals_num[2].astype(str)
        #self.num_attributive_relations_txt = len(self.attr_relations_txt)
        #nlp = spacy.load('en_core_web_md')

        # todo implement to use the non negative filter features_num_mask
        #self.features_txt = []
        #for i in tqdm(range(len(self.entities))):
        #    df_i = df_literals_txt[df_literals_txt[0] == i]

        #    features_txt_i = torch.zeros(self.num_attributive_relations_txt, 300)
        #    for index, row in df_i.iterrows():
        #        spacy_sentence = torch.tensor(nlp(row[2]).vector)
        #        features_txt_i[int(row[1])] = spacy_sentence

        #    self.features_txt.append(features_txt_i)

        #self.features_txt = torch.stack(self.features_txt)


if __name__ == '__main__':
    dataset = LiteralLinkPredDataset('./data/fb15k-237')