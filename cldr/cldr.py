import random
import pathlib
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from .model import MultiHeadAE
from .loss import TripletMarginLoss, SWD
from .augment import FactorwiseAugmentation, dsprites_transforms
from .evaluation import compute_metrics
from . import utils



def train(args):
    dataloader = utils.get_loader(
        args.dataset, 
        batch_size=args.batch_size, 
        seed=args.seed, 
        drop_last=True)
    
    if args.dataset == 'dsprites_full':
        transforms = dsprites_transforms
    augment = FactorwiseAugmentation(transforms)
    
    if args.activation == 'relu':
        activation = nn.ReLU
    
    if args.model == 'autoencoder':
        model = MultiHeadAE(args.nf, 
                            args.dpf, 
                            args.head_layers,
                            args.decoder, 
                            activation, 
                            args.bn, 
                            args.init_mode).to(device)
    model.train()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.decay)
    
    alignment = TripletMarginLoss(args.nf, args.dpf)
    distribution = SWD(sampler=lambda z: torch.randn_like(z))
    
    step = 0
    while step < args.steps:
        for x1, y1 in dataloader:
            x2, f = augment(x1)
            y = torch.arange(x1.shape[0])
            
            x = torch.cat([x1, x2]).to(device)
            f = torch.cat([f, f]).to(device)
            y = torch.cat([y, y]).to(device)
            
            x, z, xr = model.autoencode(x)
            
            if xr is not None:
                recons_loss = F.binary_cross_entropy(xr, x)
            else:
                recons_loss = torch.zeros(1).to(device)
            
            align_loss = alignment(z, y, f)
            
            dist_loss = distribution(z)
            
            loss = args.alpha*recons_loss + args.beta*dist_loss + args.gamma*align_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % args.log_interval == 0:
                print('Iteration {}: loss={}'.format(step, loss.item()))
                writer.add_scalar('Loss/Total', loss.item(), global_step=step)
                writer.add_scalar('Loss/Reconstuction', recons_loss.item(), global_step=step)
                writer.add_scalar('Loss/Distribution', dist_loss.item(), global_step=step)
                writer.add_scalar('Loss/Alignment', align_loss.item(), global_step=step)
                writer.flush()
                
            if step >= args.steps: 
                break
    
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
    description='(CLDR)'
)

# general parameters
parser.add_argument('--config', 
                    type=str, default=None, 
                    help='Config file (default: None).')
parser.add_argument('--experiment', 
                    type=str, default='debug', 
                    help='Experiment name (default: "debug").')
parser.add_argument('--model', 
                    type=str, default='autoencoder', 
                    help='Model name (default: "autoencoder").')
parser.add_argument('--dataset', 
                    type=str, default='dsprites_full', 
                    help='Dataset name (default: "dsprites_full").')
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
# model parameters
train_parser.add_argument('--nf',  
                          type=int, default=5, 
                          help='Number of factors (default: 5).')
train_parser.add_argument('--dpf',  
                          type=int, default=2, 
                          help='Dimension per factor (default: 2).')
train_parser.add_argument('--head-layers',  default=[],
                          type=int, nargs='+', 
                          help='Head hidden layers as a list of ints (default: []).')
train_parser.add_argument('--decoder',
                          action='store_true', default=False,
                          help='Use decoder for reconstruction loss.')
train_parser.add_argument('--activation', 
                          type=str, default='relu', 
                          help='Activation function (default: "relu").')
train_parser.add_argument('--bn',
                          action='store_true', default=False,
                          help='Use batch normalization.')
train_parser.add_argument('--init-mode', 
                          type=str, default=None, 
                          help='Weight initialization (default: None).')
# training parameters
train_parser.add_argument('--batch-size',  
                          type=int, default=64, 
                          help='Batch size (default: 64).')
train_parser.add_argument('--steps',  
                          type=int, default=100000, 
                          help='Number of training steps (iterations) (default: 100000).')
train_parser.add_argument('--lr',
                          type=float, default=0.001,
                          help='Learning rate (default: 0.001).')
train_parser.add_argument('--decay',
                          type=float, default=0.0,
                          help='Weight decay (default: 0.0).')
# loss function parameters
train_parser.add_argument('--alpha',
                          type=float, default=1.0,
                          help='Alpha parameter (default: 1.0).')
train_parser.add_argument('--beta',
                          type=float, default=1.0,
                          help='Beta parameter (default: 1.0).')
train_parser.add_argument('--gamma',
                          type=float, default=1.0,
                          help='Gamma parameter (default: 1.0).')
# other
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
    if args.config:
        with open(args.config) as config_file:
            configs = [line.split() for line in config_file.read().splitlines() if line]
        for config in configs:
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
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.benchmark = False
#             torch.set_deterministic(True)
            
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
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
#         torch.set_deterministic(True)
        
        device = torch.device("cuda" if args.cuda else "cpu")
        writer = SummaryWriter(log_dir=log_dir)

        args.func(args)
