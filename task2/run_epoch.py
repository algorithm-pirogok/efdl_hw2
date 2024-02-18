from enum import Enum
from time import time
import os

import numpy as np
import torch
import torch.nn as nn

from dataset import load_vocabulary, max_tokens, BrainDataset, BigBrainDataset, collate_fn
from transformer import PositionalEncoding
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASET_PATH = os.path.join(os.getcwd(), "data")

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


class GPT2(nn.Module):
    def __init__(self, num_embeddings, d_model=1024, nhead=8):
        super().__init__()
        self.encoder = nn.Sequence(
            nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=d_model,
            ),
            PositionalEncoding(
                d_model=d_model,
            )
        )
        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead)
    
    def forward(self, x: torch.Tensor, x_mask):
        encoder = self.encoder(x)
        return self.decoder(encoder, encoder, tgt_key_padding_mask=x_mask, memory_key_padding_mask=x_mask)


def get_gpt2_model(num_embeddings) -> torch.nn.Module:
    return GPT2(num_embeddings)


def run_epoch(data_mode: DataMode) -> None:
    device = torch.device("cuda:0")
    
    dataset = load_vocabulary(DATASET_PATH, "dataset_small.pt")
    gpt = get_gpt2_model(max_tokens)
    
    if data_mode is DataMode.BRAIN:
        mode = "Brain"
        collator = lambda x: collate_fn(x, None)
        loader = DataLoader(BrainDataset(DATASET_PATH), batch_size=32, collate_fn=collator)
    elif data_mode is DataMode.BIG_BRAIN:
        mode = "BigBrain"
        loader = DataLoader(BigBrainDataset(DATASET_PATH), batch_sampler=32, collate_fn=collate_fn)
    else:
        pass
    
    print("START EPOCH")
    lst = []
    for data, mask in tqdm(loader):
        torch.cuda.synchronize()
        start = time()
        gpt(start, mask)
        torch.cuda.synchronize()
        delta = time() - start
        lst.append(delta)
    
    lst = np.array(lst[5:]) # warm_up
    print(f"Mode: {mode}\nmin: {np.min(lst)}\nmean: {np.mean(lst)}\nmax: {np.max(lst)},median: {np.median(lst)}")
    
if __name__ == "__main__":
    run_epoch(DataMode.BIG_BRAIN)