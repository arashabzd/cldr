import torch
from torch import nn
from torch.nn import functional as F

from nflows import flows, distributions, transforms
from nflows.nn import nets


class ConditionalNormalRealNVP(flows.Flow):
    def __init__(
        self,
        dim,
        num_layers,
        embedding_net,
        embedding_dim,
        blocks_per_layer=2,
        activation=F.relu,
        dropout=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        mask = torch.ones(dim)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=out_features,
                context_features=embedding_dim,
                num_blocks=blocks_per_layer,
                activation=activation,
                dropout_probability=dropout,
                use_batch_norm=batch_norm_within_layers
            )

        layers = []
        for _ in range(num_layers):
            transform = transforms.AffineCouplingTransform(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(transforms.BatchNorm(features=dim))
        
        context_encoder = nets.ResidualNet(
            in_features=embedding_dim,
            out_features=2*dim,
            hidden_features=embedding_dim,
            num_blocks=blocks_per_layer,
            activation=activation,
            dropout_probability=dropout,
            use_batch_norm=batch_norm_within_layers
        )
        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distributions.ConditionalDiagonalNormal([dim], context_encoder),
            embedding_net=embedding_net
        )
