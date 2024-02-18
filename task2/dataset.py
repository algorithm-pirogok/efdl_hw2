from collections import defaultdict
import os
from typing import Optional
import pathlib
from random import shuffle

from datasets import load_dataset
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
import torchtext

MAX_LENGTH = 640
max_tokens = 5000
BASE_PATH = pathlib.Path(".")

    
def load_vocabulary(data_path, mode: str):
    dir_path = data_path / "data"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
        print("LOAD DATASET")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", ignore_verifications=True)["text"]
        
        print("CREATE TOKENS")
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        tokens = [tokenizer(elem) for elem in dataset]
        
        print("CREATE VOCABULARY")
        vocabulary = torchtext.vocab.build_vocab_from_iterator(
            iter(tokens), max_tokens=max_tokens, specials=["<unk>"]
        )
        vocabulary.set_default_index(vocabulary["<unk>"])
        
        print("FILTER")
        filter_data = [torch.tensor(vocabulary(elem)).long() for elem in tokens if len(vocabulary(elem))]
        torch.save(filter_data, dir_path / "dataset.pt")
        torch.save(filter_data[:10000], dir_path / "dataset_small.pt")
    return torch.load(dir_path / mode)
        

class BrainDataset(Dataset):
    def __init__(self, data_path, mode="dataset_small.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class BigBrainDataset(Dataset):
    def __init__(self, data_path, mode="dataset_small.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, mode="dataset_small.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx][: self.max_length]
    
    def __len__(self):
        return len(self.dataset)



def collate_fn_for_sequence(batch: list[torch.Tensor], max_length: Optional[int] = MAX_LENGTH) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: padded sequences and their masks
    """
    # Clip to maximum length
    batch = [b[:max_length] for b in batch]
    # Calculate length of output batch
    result_length = max([len(b) for b in batch])
    # Construct new batch with mask
    result = torch.zeros((len(batch), result_length), dtype=batch[0].dtype)
    mask = torch.zeros_like(result, dtype=torch.bool)
    for i, tensor in enumerate(batch):
        result[i, : tensor.size(0)] = tensor
        mask[i, : tensor.size(0)] = 1
    return result, mask


class UltraDuperBigBrainBatchSampler(Sampler):
    def __init__(self, dataset: UltraDuperBigBrainDataset, k: int, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        lengths = [len(x) for x in dataset]
        groups = defaultdict(list)
    
        # group by lengths into buckets
        for i, l in enumerate(lengths):
            assert l > 0
            groups[min(l, max_length) // k].append(i)
        print(f"{len(groups)} groups:")
        for g, indices in groups.items():
            print(f"[{g * k}, {g * k + 1}):\t{len(indices)}")

        # construct batch indices
        self.batches = []
        for g, indices in groups.items():
            # shuffle indices
            shuffle(indices)
            for b_start_ind in range(0, len(indices), batch_size):
                self.batches.append(indices[b_start_ind : b_start_ind + batch_size])
 
        # shuffle batches
        shuffle(self.batches)
        assert sum((len(b) for b in self.batches)) == len(dataset)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)