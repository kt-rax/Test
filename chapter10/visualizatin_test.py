# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y, params=dict(model.named_parameters()),show_attrs=True, show_saved=True).render("Model_Test3", format="png")