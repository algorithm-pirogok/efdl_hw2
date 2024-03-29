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
    def __init__(self, data_path, mode="dataset.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class BigBrainDataset(Dataset):
    def __init__(self, data_path, mode="dataset.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, mode="dataset.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_vocabulary(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)



def collate_fn(
    batch: list[torch.Tensor], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    if max_length is None:
        max_length = max([elem.shape[0] for elem in batch])
    answer = torch.zeros((len(batch), max_length), dtype=batch[0].dtype)
    target = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for ind, elem in enumerate(batch):
        fix_elem = elem[:max_length]
        fix_len = fix_elem.shape[0]
        answer[ind][:fix_len], target[ind][:fix_len] = fix_elem, True
    return answer, target.transpose(0, 1)


class UltraDuperBigBrainBatchSampler(Sampler):
    
    def __init__(self, dataset, k, batch_size, max_length=None):
        len_dt = defaultdict(list)
        
        for ind, elem in enumerate(dataset):
            len_dt[len(elem) // k].append(min(ind, max_length) if max_length else ind)
        
        self.batch = []
        for idx_lists in len_dt.values():
            idx_lists = torch.randperm(len(idx_lists)).tolist()
            for idx in range(0, len(idx_lists), batch_size):
                self.batch.append(idx_lists[idx: idx+batch_size])
        shuffle(self.batch)
        
    def __len__(self):
        return len(self.batch)

    def __iter__(self):
        for batch in self.batch:
            yield batch