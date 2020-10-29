import random
import pathlib
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as TF

from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import distances, miners, losses

from .models import DeterministicModel
from .models import GaussianModel
from .models import Discriminator
from .evaluation import compute_metrics
from .utils import utils


def train(args):
    print(f'Loading dataset: {args.dataset}')
    dataloader = utils.get_loader(args.dataset, 
                                  batch_size=args.batch_size, 
                                  seed=args.seed, 
                                  drop_last=True)
    
    color_jitter = TF.ColorJitter(.8*args.jitter, 
                                  .8*args.jitter, 
                                  .8*args.jitter, 
                                  .2*args.jitter)
    augment = TF.Compose(
        [
            TF.ToPILImage(), 
            TF.RandomApply([color_jitter], p=0.8), 
            TF.RandomResizedCrop(64, scale=(.5, 1)), 
            TF.ToTensor()
        ]
    )
    augment = utils.BatchTransform(augment)
    
    print(f'Initializing model: {args.model}')
    if args.model == 'deterministic':
        model = DeterministicModel(bn=args.bn).to(device)
    elif args.model == 'gaussian':
        model = GaussianModel(bn=args.bn).to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    miner = miners.MultiSimilarityMiner(epsilon=args.eps)
    ntxent = losses.NTXentLoss(temperature=args.temp)
    
    if args.tc > 0:
        discriminator = Discriminator().to(device)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.5, .9))
    
    print('Training model:')
    step = 0
    while step < args.steps:
        for x, y in dataloader:
            x1 = augment(x)
            x2 = augment(x)
            x = torch.cat([x1, x2], dim=0).to(device)
            z, u, dist = model.project(x)
            
            if args.supervise > -1:
                y = y[:, args.supervise]
            else:
                y = torch.arange(x1.shape[0])
            y = torch.cat([y, y], dim=0).to(device)
            
            pairs = miner(u, y)
            a1, p, a2, n = pairs
            
            contrastive_loss = ntxent(u, y, pairs)
            loss = contrastive_loss
            
            entropy = dist.entropy().sum(dim=1).mean()
            if args.entropy > 0 :
                loss += args.entropy * -entropy

            p_pos = torch.distributions.Normal(dist.mean[a1], dist.stddev[a1])
            q_pos = torch.distributions.Normal(dist.mean[p], dist.stddev[p])
            kl_pos = torch.distributions.kl_divergence(p_pos, q_pos).sum(dim=1).mean()
            
            p_neg = torch.distributions.Normal(dist.mean[a2], dist.stddev[a2])
            q_neg = torch.distributions.Normal(dist.mean[n], dist.stddev[n])
            kl_neg = torch.distributions.kl_divergence(p_neg, q_neg).sum(dim=1).mean()
            
            kl = (kl_pos - 0.0001 * kl_neg)
            kl = kl_pos
            if args.kl > 0:
                loss += args.kl * kl
                
            if args.tc > 0:
                dz = discriminator(z)
                tc = (dz[:, 0] - dz[:, 1]).mean()
                loss += args.tc * tc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.tc > 0:
                ones = torch.ones(z.shape[0], dtype=torch.long, device=device)
                zeros = torch.zeros(z.shape[0], dtype=torch.long, device=device)
                dz = discriminator(z.detach())
                zperm = utils.permute_latent(z.detach())
                dzperm = discriminator(zperm)
                discriminator_loss = 0.5*(F.cross_entropy(dz, zeros) + F.cross_entropy(dzperm, ones))
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()
            
            step += 1
            
            if step % args.log_interval == 0:
                print('Iteration {}: loss={}'.format(step, loss.item()))
                
                writer.add_scalar('Pairs/Positive', a1.shape[0], global_step=step)
                writer.add_scalar('Pairs/Negative', a2.shape[0], global_step=step)
                writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), global_step=step)
                writer.add_scalar('Loss/Entropy', entropy.item(), global_step=step)
                writer.add_scalar('Loss/Total', loss.item(), global_step=step)
                writer.add_scalar('KL-Divergence/Positive', kl_pos.item(), global_step=step)
                writer.add_scalar('KL-Divergence/Negative', kl_neg.item(), global_step=step)
                
                if args.tc > 0:
                    writer.add_scalar('TotalCorrelation', tc.item(), global_step=step)
                writer.flush()
                
            if step >= args.steps: 
                break
    
    print('Exporting model')
    utils.export_model(model, model_path)


def evaluate(args):
    args.metrics = None if args.metrics == '' else args.metrics.split()
    compute_metrics(model_path=model_path, 
                    output_dir=evaluation_dir, 
                    dataset_name=args.dataset, 
                    cuda=args.cuda, 
                    seed=args.seed,
                    metrics=args.metrics)
    

def visualize(args):
    dataset = utils.DLIBDataset(args.dataset, seed=args.seed)
    model = utils.import_model(model_path).to(device)
    y, x = dataset.dataset.sample(args.sample_size, random_state=dataset.random_state)
    x = torch.from_numpy(np.moveaxis(x, 3, 1))
    with torch.no_grad():
        r = model(x.to(device))
    writer.add_embedding(
        r,
        metadata=[zi for zi in z[:, args.label]],
        label_img=x,
    )
    writer.flush()
    


parser = argparse.ArgumentParser(
    description='Contrastive Learning of Disentangled Representations (CLDR)'
)
parser.add_argument('--config', 
                    type=str, default='', 
                    help='Config file (default: "").')
parser.add_argument('--experiment', 
                    type=str, default='debug', 
                    help='Experiment name (default: "debug").')
parser.add_argument('--model', 
                    type=str, default='gaussian', 
                    help='Model name (default: "gaussian").')
parser.add_argument('--dataset', 
                    type=str, default='cars3d', 
                    help='Dataset name (default: "cars3d").')
parser.add_argument('--seed', 
                    type=int, default=0,
                    help='Random seed (default: 0).')
parser.add_argument('--cuda',
                    action='store_true', default=False,
                    help='Enable CUDA.')

subparsers = parser.add_subparsers(
    help=''
)
train_parser = subparsers.add_parser(
    'train',
    help='Train a model.'
)
train_parser.add_argument('--bn',
                          action='store_true', default=False,
                          help='Enable Batch Normalization.')
train_parser.add_argument('--jitter',
                          type=float, default=0.5,
                          help='Color jitter (default: 0.5).')
train_parser.add_argument('--batch-size',  
                          type=int, default=64, 
                          help='Batch size (default: 64).')
train_parser.add_argument('--steps',  
                          type=int, default=100000, 
                          help='Number of training steps (iterations) (default: 100000).')
train_parser.add_argument('--lr',
                          type=float, default=0.0005,
                          help='Learning rate (default: 0.0005).')
train_parser.add_argument('--decay',
                          type=float, default=0.0,
                          help='Weight decay (default: 0.0).')
train_parser.add_argument('--eps',
                          type=float, default=0.1,
                          help='Epsilon parameter of MultiSimilarityMiner (default: 0.1).')
train_parser.add_argument('--temp',
                          type=float, default=0.07,
                          help='Temperature parameter (default: 0.07).')
train_parser.add_argument('--entropy',
                          type=float, default=0.0,
                          help='Entropy regularization strength (default: 0.0).')
train_parser.add_argument('--kl',
                          type=float, default=0.0,
                          help='KL-Divergence regularization strength (default: 0.0).')
train_parser.add_argument('--tc',
                          type=float, default=0.0,
                          help='TotalCorrelation regularization strength (default: 0.0).')
train_parser.add_argument('--supervise', 
                          type=int, default=-1,
                          help='Index of factor used for supervision (default: -1).')
train_parser.add_argument('--log-interval', 
                          type=int, default=10,
                          help='Tensorboard log interval (default: 10).')
train_parser.set_defaults(func=train)

eval_parser = subparsers.add_parser(
    'evaluate',
    help='Evaluate a model.'
)
eval_parser.add_argument('--metrics', 
                         type=str, default='', 
                         help='List of metrics (default: All available metrics).')
eval_parser.set_defaults(func=evaluate)

vis_parser = subparsers.add_parser(
    'visualize',
    help='Make a visualization of model representations in tensorboard.'
)
vis_parser.add_argument('--sample-size',  
                        type=int, default=200, 
                        help='Sample size (default: 200).')
vis_parser.add_argument('--label',  
                        type=int, default=0, 
                        help='Factor used as label (default: 0).')
vis_parser.set_defaults(func=visualize)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config != '':
        with open(args.config) as config_file:
            runs = config_file.readlines()
        for run in runs:
            config = run.split()
            args = parser.parse_args(config)
            result_dir = pathlib.Path('./results/')/args.experiment/args.model/args.dataset
            log_dir = result_dir/'log'
            model_dir = result_dir/'saved_model'
            evaluation_dir = result_dir/'evaluation'
            log_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            evaluation_dir.mkdir(parents=True, exist_ok=True)

            log_dir = str(log_dir)
            evaluation_dir = str(evaluation_dir)
            model_path = str(model_dir/'model.pt')

            random.seed(args.seed)
            torch.manual_seed(args.seed)
            device = torch.device("cuda" if args.cuda else "cpu")
            writer = SummaryWriter(log_dir=log_dir)
            
            args.func(args)
    else:
        result_dir = pathlib.Path('./results/')/args.experiment/args.model/args.dataset
        log_dir = result_dir/'log'
        model_dir = result_dir/'saved_model'
        evaluation_dir = result_dir/'evaluation'
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        log_dir = str(log_dir)
        evaluation_dir = str(evaluation_dir)
        model_path = str(model_dir/'model.pt')

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if args.cuda else "cpu")
        writer = SummaryWriter(log_dir=log_dir)

        args.func(args)
