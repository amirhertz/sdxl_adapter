from __future__ import  annotations
from dataclasses import dataclass
import enum
from custom_types import *


class ModelType(enum.Enum):
    Adapter = 'adapter'


@dataclass(frozen=True)
class Region:

    corner_y: int  # for top left corner
    corner_x: int  # for top left corner
    height: int
    width: int
    is_reference: bool
    is_aligned: bool
    crop_ratio_min: float
    random_flip: bool
    train_images_dir: str
    eval_images_dir: str = ''

    @property
    def top(self):
        return self.corner_y

    @property
    def left(self):
        return self.corner_x

    @property
    def bottom(self):
        return self.corner_y + self.height

    @property
    def right(self):
        return self.corner_x + self.width


@dataclass
class DeltaNetworkArgs:
    num_hidden_layers: int = 5
    hidden_dim: int = 128
    is_normalized: bool = False


@dataclass
class BaseArgs:
    lr: float = 1e-4
    batch_size = 2
    warmup_steps: int = 5000
    num_training_steps: int = 200000
    save_checkpoint_every: int = 10000


@dataclass
class ArgsAdapter(BaseArgs):
    @property
    def text_prompt(self) -> str:
        return f'{self.content_desc} {self.style_desc}'

    image_path: str | tuple[str, ...]  = ''
    save_checkpoint_every: int = 500
    show_every: int = 50
    region: Region = Region(0, 0, 1024, 1024, False, False, 1., False, '')
    rank: int = 1
    batch_size: int = 1
    model_type: ModelType = ModelType.Adapter
    style_desc: str = ''
    content_desc:str = ''
    eval_prompts: tuple[str, ...] = ()
