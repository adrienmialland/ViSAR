
from PIL.Image import Image
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def from_pretrained(self):
        pass

    @abstractmethod
    def generate(self, query: str, images: list[Image], no_parse=False):
        pass
