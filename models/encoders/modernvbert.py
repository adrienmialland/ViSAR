
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

# Filter the following torch.load warning that uselessly flood the terminal (from pickle.load in _load_embed_doc_images):
# # You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. 
# # It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). 
# # In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. 
# # Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. 
# # We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
from transformers.utils import logging

from data import Dataset
from models.encoders import BaseEncoder

# Implementation of llava-onevision-qwen2-7b model using 
# the publicly available Hugging Face APIs. 
#
# see: https://huggingface.co/ModernVBERT/colmodernvbert
class ColModern_VBert(BaseEncoder):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.processor = None

        self.BATCH_SIZE = 2
    
        torch.set_float32_matmul_precision('high')

    def from_pretrained(self):
        if self.model == None:
            self.model = ColModernVBert.from_pretrained(
                "ModernVBERT/colmodernvbert", dtype=torch.float32, device_map="auto", trust_remote_code=True, revision='1f247b4098c641593b6ba2e64756460b7a48a126'
            )
            self.processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert", revision='3bcd984f8f83c96dccb76cf2ed6d6a8108d3a046')

    def encode_dataset(self, dataset: Dataset):
        embed_dir: Path = Path(dataset.embedding_dir).joinpath(self.name)
        embed_dir.mkdir(exist_ok=True, parents=True)
        print(f"\nEncoding document pages in {embed_dir}\nNote: already extracted screenshots are ignored\n")

        uniq_pdf_names = sorted(set([
            sample['doc_id'] for sample in dataset.load_data()
        ]), key=lambda d: d)

        for name in tqdm(uniq_pdf_names):
            embed_doc_file = embed_dir.joinpath(name + '.pt')
            if embed_doc_file.exists():
                continue
            self.from_pretrained()

            images = dataset.load_sample_images(name)
            
            logging.set_verbosity_error()
            embed_doc_imgs = self.encode_images(images)
            logging.set_verbosity_warning()

            torch.save(embed_doc_imgs, embed_doc_file)

    def encode_images(self, images: list):
        collate_fn = lambda images: self.processor.process_images(images)
        dataloader = DataLoader(images, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        embed_doc_imgs = []
        for batch_img in dataloader:
            with torch.no_grad():
                embed_doc_imgs.extend(
                    self.model(**{
                        k: v.to(self.model.device) for k, v in batch_img.items()
                    })
                )

        return embed_doc_imgs

    def encode_query(self, query: str):
        with torch.no_grad():
            embed_query = self.model(
                **self.processor.process_texts([query]).to(self.model.device)
            )
        
        return embed_query

    def compute_scores(self, embed_pages: list[torch.Tensor], embed_query: torch.Tensor) -> list:
        scores = self.processor.score(embed_query, embed_pages)
        return scores[0].tolist()
