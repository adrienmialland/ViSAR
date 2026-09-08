
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

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

# Common implementation shared by Vidore encoders using the publicly
# available Hugging Face APIs. See the individual encoder classes
# below for encoder-specific behavior and the corresponding
# Hugging Face model links.
class ViDoReEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()

        self.model = None
        self.processor = None

        self.BATCH_SIZE = 2
    
    def from_pretrained(self):
        "Defined Below"

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
        collate_fn = lambda images: self.processor.process_images(images).to(self.model.device)
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
                **self.processor.process_queries([query]).to(self.model.device)
            )

        return embed_query

    def compute_scores(self, embed_pages: list[torch.Tensor], embed_query: torch.Tensor) -> list:
        scores = self.processor.score_multi_vector(embed_query, embed_pages)
        return scores[0].tolist()

# see: https://huggingface.co/vidore/colpali-v1.2
class Colpali_v1_2(ViDoReEncoder):
    def __init__(self):
        super().__init__()

    def from_pretrained(self):
        if self.model == None:
            self.model = ColPali.from_pretrained(
                "vidore/colpali-v1.2", revision='30ab955d073de4a91dc5a288e8c97226647e3e5a', dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(
                "vidore/colpali-v1.2", revision='6b89bc63c16809af4d111bfe412e2ac6bc3c9451'
            )
            # For reproducibility due to colpali_engine version.
            # We currently use colpali_engin 0.3.13, while the embeddings that we generated 
            # were obtained with a previous version that used a different 'visual_prompt_prefix', while 
            # using the same model weights. Therefore, the following reproduces the embedings that we generated.
            self.processor.visual_prompt_prefix = "<image>Describe the image." # Only used for image encoding

# see: https://huggingface.co/vidore/colqwen2.5-v0.1
class ColQwen2_5_v0_1(ViDoReEncoder):
    def __init__(self):
        super().__init__()

    def from_pretrained(self):
        if self.model == None:
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            self.model = ColQwen2_5.from_pretrained(
                "vidore/colqwen2.5-v0.1", revision='92908120384b7a2110c5beda3ab29cbdb2c08e49', dtype=torch.bfloat16, device_map="auto", attn_implementation=attn_impl
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(
                "vidore/colqwen2.5-v0.1", revision='9b8ddce7a35cb55148502554d8d640d8b499930a', use_fast=True
            )

