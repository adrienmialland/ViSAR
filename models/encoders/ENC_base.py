
import gc
import torch
from pathlib import Path
from abc import ABC, abstractmethod

from data import Dataset

class BaseEncoder(ABC):
    def __init__(self):
        super().__init__()
        
        self.name = None

    @abstractmethod
    def from_pretrained(self):
        pass

    @abstractmethod
    def encode_dataset(self, dataset: Dataset):
        pass

    @abstractmethod
    def encode_query(self, query: str):
        pass

    @abstractmethod
    def compute_scores(self, embed_pages: list[torch.Tensor], embed_query: torch.Tensor) -> list:
        pass

    def load_embed_pages(self, file_name: str, embed_dir):
        embed_dir: Path = Path(embed_dir).joinpath(self.name)
        embed_doc_file = embed_dir.joinpath(file_name + '.pt')

        return torch.load(embed_doc_file)

    def empty_memory(self):
        del self.model
        self.model = None

        del self.processor
        self.processor = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()