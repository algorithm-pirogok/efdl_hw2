from enum import Enum
from time import time
import os

import numpy as np
import torch
import torch.nn as nn

from dataset import load_vocabulary, max_tokens, BrainDataset, BigBrainDataset, collate_fn, BASE_PATH
from transformer import PositionalEncoding
from torch.utils.data import DataLoader
from tqdm import tqdm

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


class GPT2(nn.Module):
    def __init__(self, num_embeddings, d_model=1024, nhead=8):
        super().__init__()
        self.encoder = nn.Sequential(
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
    
    gpt = get_gpt2_model(max_tokens).to(device)
    
    if data_mode is DataMode.BRAIN:
        mode = "Brain"
        collator = lambda x: collate_fn(x, None)
        dataset = BrainDataset(BASE_PATH)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collator, sampler=None)
    elif data_mode is DataMode.BIG_BRAIN:
        mode = "BigBrain"
        dataset = BigBrainDataset(BASE_PATH)
        collator = lambda x: collate_fn(x, None)
        loader = DataLoader(data_mode, batch_sampler=32, collate_fn=collator, sampler=None)
    else:
        pass
    
    print("START EPOCH")
    lst = []

    for data, mask in tqdm(loader):
        torch.cuda.synchronize()
        start = time()
        gpt(data.to(device), mask.to(device))
        torch.cuda.synchronize()
        delta = time() - start
        lst.append(delta)
    
    lst = np.array(lst[5:]) # warm_up
    print(f"Mode: {mode}\nmin: {np.min(lst)}\nmean: {np.mean(lst)}\nmax: {np.max(lst)}\nmedian: {np.median(lst)}")
    
if __name__ == "__main__":
    run_epoch(DataMode.BIG_BRAIN)