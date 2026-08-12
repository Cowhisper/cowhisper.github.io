---
title: 'YACRS: Yet Another Configuration and Registration System'
published: 2026-08-07
draft: false
tags: ['projects']
toc: true
---

## Intro

**YACRS** is a configuration and registration system for Python. Providing a simple and consistent way to define and use hyperparameters. It named after **YACS** (Yet Another Configuration System), which is a widely used configuration system. 


## Why YACRS ?


I used to struggle with reading projects when i was a rookie algo engineer. I found it difficult to locates where the mappings between hyperparameters and their values were defined. Hyperparameters are always defined in a separate configuration file. Whereas, finding how it was used in the codebase is not always a easy one, especially when the codebase is large. Eg. This is what most projects do:

```python
# keep all hyperparameters in a global config class
from xxx import cfg

# The first option is to define a class or function taking the cfg as an argument.

class Model(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = torch.nn.Linear(cfg.model.input_dim, cfg.model.output_dim)

    def forward(self, x):
        return self.net(x)


def train(cfg):
    # use cfg to access hyperparameters
    lr = cfg.lr
    optimizer_zoo = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW,
    }
    optimizer = optimizer_zoo[cfg.optimizer](lr=lr)
    model = Model(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()

train(cfg)

# The second option is to map the cfg hyperparameters to the codebase yourself.
from xxx import cfg

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.net = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


def train(lr, optimizer, input_dim, output_dim):
    optimizer_zoo = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW,
    }
    optimizer = optimizer_zoo[optimizer](lr=lr)
    model = Model(input_dim, output_dim)
    loss_fn = torch.nn.CrossEntropyLoss()

train(
    lr=cfg.lr,
    optimizer=cfg.optimizer,
    input_dim=cfg.model.input_dim,
    output_dim=cfg.model.output_dim,
)
```

There are pros and cons in both approaches. The first option, taking cfg as an argument, results in cleaner code but requires passing cfg to every function. It means no one knows how many hyperparameters the class or function rely on if they do not read the whole code definition. The second option, mapping cfg hyperparameters to the codebase yourself, is more verbose but does not require passing cfg to every function. To be honest, I would say it makes the codebase complicated and redundant. This is where YACRS comes in.


## What does YACRS do?

```python
from typing import Annotated
from yacrs import _C, cli, configurable, Node, register
import torch

# _C is a global config object
_C.register("model")
_C.model.input_dim = 10
_C.model.output_dim = 10
_C.register("train")
_C.train.optimizer = "Adam"
_C.Adam = Node()
_C.Adam.lr = 0.001

register("Adam")(torch.optim.Adam)
register("SGD")(torch.optim.SGD)
register("AdamW")(torch.optim.AdamW)


@register("model")
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.net = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


@register("train")
def train(optimizer):
    model = Model()  # the argument mapping is performed automatically 
    optimizer = configurable()(optimizer)(model.parameters())

train()
```

In most cases, hyperparameters are configured in the config file (probably `yaml` or `toml` files). So we offer a convenient way to integrate them into your code.

```python
@cli()
@register("train")
def train(optimizer):
    model = Model()  # the argument mapping is performed automatically 
    optimizer = configurable()(optimizer)(model.parameters())

train()
```
By wrapping the `train` function with `@cli()` , you can use `-c` option in CLI to specify the config file and override default values and values from the config file by passing command line arguments.
Eg.
```bash
python train.py -c config.yaml train.optimizer=SGD SGD.lr:float=0.001
```
