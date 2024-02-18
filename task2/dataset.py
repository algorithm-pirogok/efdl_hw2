import os
from typing import Optional
import pathlib

from datasets import load_dataset
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
import torchtext

MAX_LENGTH = 640
max_tokens = 5000
BASE_PATH = pathlib.Path("data")

    
def load_vocabulary(data_path, mode: str):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
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
        torch.save(filter_data, BASE_PATH / "dataset.pt")
        torch.save(filter_data[:10000], BASE_PATH / "dataset_small.pt")
    return torch.load(BASE_PATH / mode)
        

class BrainDataset(Dataset):
    def __init__(self, data_path, mode="dataset_small.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_dataset(data_path, mode)
        self.max_length = max_length
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]


class BigBrainDataset(Dataset):
    def __init__(self, data_path, mode="dataset_small.pt", max_length: int = MAX_LENGTH):
        self.dataset = load_dataset(data_path, mode)
        self.max_length = max_length

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, mode="dataset_small.pt", max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


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
    answer = torch.zeros((batch.shape[0], max_length), dtype=batch[0].dtype)
    target = torch.zeros((batch.shape[0], max_length), dtype=batch[0].dtype)
    for ind, elem in enumerate(batch):
        fix_elem = elem[:max_length]
        fix_len = fix_elem.shape[0]
        answer[ind][:fix_len], target[ind][:fix_len] = fix_elem, 1
    return answer, target


class UltraDuperBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
