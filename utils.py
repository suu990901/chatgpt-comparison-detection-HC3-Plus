import torch 
from torch import nn
from torch import optim as optim
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
import os
from pathlib import Path

def create_optimizer(args, model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(parameters, **opt_args)
    return optimizer
