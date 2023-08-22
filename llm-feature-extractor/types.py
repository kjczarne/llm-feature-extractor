import torch
from torch import nn
from typing import Literal

Model = nn.Module
Property = str
Prompt = str
ModelId = str
SupportedStrategies = Literal["prompted", "fine_tuned"]
