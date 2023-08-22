from dataclasses import dataclass
from .types import ModelId, SupportedStrategies


@dataclass
class Config:
    model_id: ModelId
    strategy: SupportedStrategies