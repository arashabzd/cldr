import argparse
import pathlib

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from pytorch_metric_learning import distances, miners, losses

from .models import DlibModel
from .utils import utils


def train(args):
    results_dir = pathlib.Path('./results/')/args.experiment/args.model/args.dataset
    logs_dir = results_dir/'logs'
    models_dir = results_dir/'saved_models'
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    writer = SummaryWriter(log_dir=str(logs_dir))
    
    jitter = .5
    dataloader = utils.get_loader(args.dataset, batch_size=args.batch_size, seed=args.seed)
    color_jitter = transforms.ColorJitter(.8*jitter, .8*jitter, .8*jitter, .2*jitter)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    augment = transforms.Compose(
        [
            transforms.ToPILImage(), 
            rnd_color_jitter, 
            rnd_gray,
            transforms.RandomResizedCrop(64), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )
    augment = utils.BatchTransform(augment)
    
    if args.model == 'dlib':
        model = DlibModel(3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    distance = distances.CosineSimilarity()
    miner = miners.MultiSimilarityMiner(epsilon=.1, distance=distance)
    ntxent = losses.NTXentLoss(temperature=.5, distance=distance)
    
    step = 0
    while step < args.steps:
        for x in dataloader:
            optimizer.zero_grad()
            x1 = augment(x).to(device)
            x2 = augment(x).to(device)
            z1 = model(x1)
            z2 = model(x2)
            z = torch.cat([z1, z2], dim=0)
            y = torch.arange(z1.shape[0])
            y = torch.cat([y, y], dim=0)
            pairs = miner(z, y)
            loss = ntxent(z, y, pairs)
            loss.backward()
            optimizer.step()
            step += 1
            
            if step % args.log_interval == 0:
                print('Iteration {}: n_pos = {}, n_neg = {}, loss={}'.format(
                step,
                pairs[0].shape[0],
                pairs[2].shape[0],
                loss.item()
                ))
                writer.add_scalar('Loss', loss.item(), global_step=step)
                writer.add_scalar('Positive Samples Pairs', pairs[0].shape[0], global_step=step)
                writer.add_scalar('Negative Samples Paris', pairs[2].shape[0], global_step=step)
            
            if step >= args.steps: 
                break
    
    utils.export_model(model, str(models_dir/'model.pt'))

    
def evaluate(args):
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Contrastive Learning of Disentangled Representations (CLDR)'
    )
    subparsers = parser.add_subparsers(
        help=''
    )
    train_parser = subparsers.add_parser(
        'train',
        help='Train a model.'
    )
    train_parser.add_argument('--experiment', 
                              type=str, default='default', 
                              help='Experiment name (default="default").')
    train_parser.add_argument('--model', 
                              type=str, default='dlib', 
                              help='Model name (default="dlib").')
    train_parser.add_argument('--dataset', 
                              type=str, default='cars3d', 
                              help='Dataset name (default="cars3d").')
    train_parser.add_argument('--batch-size',  
                              type=int, default=64, 
                              help='Batch size (default=64).')
    train_parser.add_argument('--steps',  
                              type=int, default=10000, 
                              help='Number of training steps (iterations) (default=10000).')
    train_parser.add_argument('--cuda',
                              action='store_true', default=False,
                              help='Enable CUDA training.')
    train_parser.add_argument('--seed', 
                              type=int, default=0,
                              help='Random seed (default: 0).')
    train_parser.add_argument('--log-interval', 
                              type=int, default=1,
                              help='Tensorboard log interval (default: 1).')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a model.'
    )
    eval_parser.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
