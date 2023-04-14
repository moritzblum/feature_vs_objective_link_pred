from datetime import datetime

import ray
import torch
from torch.utils.data import Dataset
import pandas as pd
import os.path as osp
import numpy as np
from tqdm import tqdm
import spacy
import time

from ray import tune, air
import ray

from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from dataset import LiteralLinkPredDataset
from models import DistMult, DistMultRegression
import os
from ray.tune import Tuner, ExperimentAnalysis, CLIReporter


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


def train_standard_lp(config,
                      model_lp,
                      model_features,
                      loss_function_model,
                      loss_function_features,
                      optimizer,
                      dataset):
    model_lp.train()
    start = time.time()

    train_edge_index_t = dataset.edge_index_train.t().to(DEVICE)
    train_edge_type = dataset.edge_type_train.to(DEVICE)
    features_num = dataset.features_num.to(DEVICE)

    edge_index_batches = torch.split(train_edge_index_t, config['batch_size'])
    edge_type_batches = torch.split(train_edge_type, config['batch_size'])

    batch_indices = np.arange(len(edge_index_batches))
    np.random.shuffle(batch_indices)

    loss_total = 0
    for batch_index in batch_indices:
        edge_idxs, relation_idx = edge_index_batches[batch_index], edge_type_batches[batch_index]
        optimizer.zero_grad()

        edge_idxs_neg = negative_sampling(edge_idxs, dataset.num_entities, eta=config['eta'])

        out_pos = model_lp.forward(edge_idxs[:, 0], relation_idx, edge_idxs[:, 1])
        out_neg = model_lp.forward(edge_idxs_neg[:, 0], relation_idx.repeat(config['eta']), edge_idxs_neg[:, 1])

        out = torch.cat([out_pos, out_neg], dim=0)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * config['eta'])], dim=0).to(
            DEVICE)

        #print('size lp:', gt.size())

        loss = loss_function_model(out, gt)

        if config['reg']:
            loss += 1e-5 * model_lp.l3_regularization()

        if config['alpha'] > 0:
            batch_entities = torch.tensor(list(set(edge_idxs[:, 0].tolist() + edge_idxs[:, 1].tolist()))).to(DEVICE)
            out_flatten = model_features.forward(model_lp.entity(batch_entities)).flatten()
            gold_flatten = features_num[batch_entities].flatten()
            # we only score triples that have an attribute value
            relevant_idx_mask = dataset.features_num_mask.to(DEVICE)[batch_entities].flatten()

            feature_loss = loss_function_features(out_flatten[relevant_idx_mask],
                                                  gold_flatten[relevant_idx_mask])

            #print('size feature:', out_flatten[relevant_idx_mask].size())

            #print('loss', loss)
            #print('feature_loss', feature_loss)
            loss = (1 - config['alpha']) * loss + config['alpha'] * (100 * feature_loss)

        loss_total += loss
        loss.backward()
        optimizer.step()
    end = time.time()
    #print('elapsed time:', end - start)
    #print('loss:', loss_total / len(edge_index_batches))


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr_triple_scoring(model_lp, dataset, eval_edge_index, eval_edge_type,
                               fast=False):
    model_lp.eval()
    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for triple_index in range(num_samples):
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

        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

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

        out = model_lp.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

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


def train_lp_objective(config):
    dataset = ray.get(dataset_ray)

    model_lp = DistMult(dataset.num_entities, dataset.num_relations, config['dim'], config['dropout'],
                        batch_norm=config['batch_norm'])
    model_lp.to(DEVICE)
    model_features = DistMultRegression(config['dim'], dataset.num_attributive_relations_num)
    model_features.to(DEVICE)

    loss_function_model = torch.nn.BCELoss(reduction='mean')
    loss_function_features = torch.nn.L1Loss(reduction='mean')
    if config['alpha'] > 0:
        optimizer = torch.optim.Adam(list(model_lp.parameters()) + list(model_features.parameters()), lr=config['lr'])
    else:
        optimizer = torch.optim.Adam(model_lp.parameters(), lr=config['lr'])

    start_epoch = 1
    # restore a checkpoint
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state_lp, model_state_features, optimizer_state, states = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model_lp.load_state_dict(model_state_lp)
        model_features.load_state_dict(model_state_features)
        optimizer.load_state_dict(optimizer_state)
        start_epoch = states['epoch']
        print('restored checkpoint with old epoch state:', start_epoch)

    model_lp.train()
    model_features.train()

    for epoch in range(start_epoch, 1000):
        train_standard_lp(config,
                          model_lp,
                          model_features,
                          loss_function_model,
                          loss_function_features,
                          optimizer,
                          dataset)
        if epoch % 50 == 0:
            mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                              dataset,
                                                                              dataset.edge_index_val,
                                                                              dataset.edge_type_val,
                                                                              fast=True)
            print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
            torch.save(model_lp.state_dict(), 'model_lp.pth')
            torch.save(model_features.state_dict(), 'model_features.pth')

            os.makedirs("model_checkpoint", exist_ok=True)
            torch.save(
                (model_lp.state_dict(), model_features.state_dict(), optimizer.state_dict(), {"epoch": epoch}),
                "model_checkpoint/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("model_checkpoint")
            session.report(
                {"mrr": mrr.item(), "mr": mr, "hits10": hits10, "hits5": hits5, "hits3": hits3, "hits1": hits1},
                checkpoint=checkpoint)
            # tune.report({"mrr": mrr.item()})


if __name__ == '__main__':
    PROJECT_DIR = '/homes/mblum/feature_vs_objective_link_pred'
    # place ~/ray_results/RUN_NAME here to resume the called RUN_NAME e.g. ~/ray_results/train_2023-02-22_12-59-53
    resume_training_from = ''  # /homes/mblum/ray_results/feature_vs_objective_link_pred_2023-02-23_10-36-24
    DEVICE = torch.device('cuda')
    RUN_NAME = 'feature_vs_objective_link_pred_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_name = 'fb15k-237'
    if not osp.isfile(osp.join(PROJECT_DIR, f'data/{dataset_name}/processed.pt')):
        dataset = LiteralLinkPredDataset(osp.join(PROJECT_DIR, f'data/{dataset_name}'))
        torch.save(dataset, osp.join(PROJECT_DIR, f'data/{dataset_name}/processed.pt'))
    print('load processed dataset')
    dataset = torch.load(osp.join(PROJECT_DIR, f'data/{dataset_name}/processed.pt'))

    # ray.init("auto")  # just required for slurm workload management otherwise causes error
    dataset_ray = ray.put(dataset)

    # search_space = {
    #    'dataset_name': dataset_name,
    #    'dim': tune.grid_search([100, 150, 200]),
    #    "lr": tune.loguniform(1e-4, 1e-2),
    #    "batch_size": tune.grid_search([128, 256, 1024]),  # 128 for completeness
    #    'alpha': tune.grid_search([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # alpha of 0 leads to regular DistMult without literal features
    #    'eta': tune.grid_search([1, 2, 5, 10, 20]),
    #    'reg': tune.grid_search([True, False]),
    #    'batch_norm': tune.grid_search([True, False]),
    #    'dropout': tune.grid_search([0, 0.1, 0.2])
    # }

    # default config
    train_lp_objective(config={'dataset_name': dataset_name,
                               'dim': 200,
                               'lr': 0.001,
                               'batch_size': 256,
                               'dropout': 0.2,
                               'alpha': 0.4,
                               'eta': 5,
                               'reg': False,
                               'batch_norm': False})

    if resume_training_from == '':
        reporter = CLIReporter(max_progress_rows=10)
        reporter.add_metric_column("mrr")

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_lp_objective),
                resources={"cpu": 4, "gpu": 0.5}  # usually two trials can share a GPU, therefore, 0.5 is enough
            ),
            tune_config=tune.TuneConfig(
                scheduler=ASHAScheduler(metric="mrr", mode="max"),
                num_samples=6,
                reuse_actors=False,
            ),
            run_config=air.RunConfig(progress_reporter=reporter,
                                     name=RUN_NAME,
                                     checkpoint_config=air.CheckpointConfig(
                                         checkpoint_score_attribute="mrr",
                                         num_to_keep=5)),
            param_space=search_space,
            _tuner_kwargs={"raise_on_failed_trial": True}
        )
    else:
        RUN_NAME = resume_training_from.split('/')[-1]
        tuner = Tuner.restore(resume_training_from, resume_unfinished=True)

    tuner.fit()
    analysis = ExperimentAnalysis(experiment_checkpoint_path=f'/homes/mblum/ray_results/{RUN_NAME}')

    # load the best performing model
    best_run_name = analysis.get_best_logdir("mrr", mode="max")
    state_dict = torch.load(os.path.join(best_run_name, "model_lp.pth"))
    print('best config:', analysis.get_best_config("mean_accuracy", mode="max"))
    print('best model file:', os.path.join(best_run_name, "model.pth"))
    best_config = analysis.get_best_config("mrr", mode="max")
    dataset = torch.load(osp.join(PROJECT_DIR, f'data/{dataset_name}/processed.pt'))
    model_lp = DistMult(dataset.num_entities, dataset.num_relations, best_config['dim'], best_config['dropout'],
                        batch_norm=best_config['batch_norm'])
    model_lp.load_state_dict(state_dict)
    model_lp.to(DEVICE)
    results = {'config': best_config, 'date': time.strftime("%Y%m%d")}
    print(results)

    mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model_lp,
                                                                      dataset,
                                                                      dataset.edge_index_test,
                                                                      dataset.edge_type_test,
                                                                      fast=True)
    print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
