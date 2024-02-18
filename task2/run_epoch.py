from enum import Enum
import os


import torch
from dataset import load_vocabulary

DATASET_PATH = os.path.join(os.getcwd(), "data")

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model() -> torch.nn.Module:
    pass


def run_epoch(data_mode: DataMode) -> None:
    pass


load_vocabulary(DATASET_PATH)