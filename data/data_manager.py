
from typing import Dict, Any, TypeAlias
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import itertools
import pymupdf
import shutil
import json
import os

from config import DatasetConfig

NestedDict: TypeAlias = Dict[str, Dict[str, Dict[str, Any]]]

NOT_EXTRACTED_ERROR = "No extracted data available. You should probably first run 'python ./script/extract.py --config <config_file>'"
NOT_ENCODED_ERROR   = "No embedding data available. You should probably first run 'python ./script/encode.py --config <config_file>'"

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_ROOT_DIR = PROJECT_ROOT / 'data'

def download_dataset(name, include_documents):
    '''
    Automaticaly downloads the datasets if not already downloaded.

    It always verify if 'sample.json' file has been downloaded, as it contains the actual question-answer samples.

    if 'include_documents' input agument is set to True, it checks if the pdf documents need to be downloaded, which is required for page screenshots extraction and page embeddings.
    '''
    from huggingface_hub import snapshot_download

    revision = '0dfdcc9a777ebf393653e4b7fc442aa89a8cfabc'

    patterns = [f'{name}/samples.json']
    if include_documents:
        patterns.append(f'{name}/documents/**')

    # Download the dataset from: 
    # https://huggingface.co/datasets/Lillianwei/Mdocagent-dataset/tree/main
    snapshot_download(
        repo_id='Lillianwei/Mdocagent-dataset',
        repo_type='dataset',
        allow_patterns=patterns,
        local_dir=DATA_ROOT_DIR,
        revision=revision
    )

class Dataset():
    def __init__(self, dataset: DatasetConfig, runtime: bool = False):
        self.name = dataset.name

        self.dataset_dir = DATA_ROOT_DIR / dataset.name
        self.documents_dir = self.dataset_dir / 'documents'
        self.extracted_dir = self.dataset_dir / 'extracted'
        self.embedding_dir = self.dataset_dir / 'embedding'
        self.results_dir   = self.dataset_dir / 'results'

        self.qas_data = Path(self.dataset_dir).joinpath(dataset.qas_data)
        self.qas_data_results = Path(self.dataset_dir).joinpath(dataset.qas_data_results)

        self.BUILD_IMG_PATH = lambda doc_name, index: f"{self.extracted_dir}/{doc_name}/{doc_name}_{index}.png"

        self.last_save = datetime.now().timestamp()
        self.new_data = False

        if runtime:
            if not self.extracted_dir.exists():
                raise FileNotFoundError(NOT_EXTRACTED_ERROR)
            if not self.embedding_dir.exists():
                raise FileNotFoundError(NOT_ENCODED_ERROR)

            if not self.qas_data.exists():
                download_dataset(self.name, False)
            
            if not self.qas_data_results.exists():
                shutil.copyfile(self.qas_data, self.qas_data_results)
        else:
            download_dataset(self.name, True)

    def load_data(self, from_results=False) -> list[NestedDict]:
        if from_results:
            assert self.qas_data_results.exists(), 'no question-answer results file was found'
            data_file = self.qas_data_results
        else:
            assert self.qas_data.exists(), 'no question-answer file was found'
            data_file = self.qas_data

        with open(data_file, 'r', encoding="utf-8") as f:
            data = json.load(f)

        return data

    def save_results(self, data, interval=0):
        if not self.new_data or datetime.now().timestamp() - self.last_save < interval:
            return None

        try:
            # to avoid loss of data if an interuption occurs during saving.
            # os.replace then overwrite qas_data_results file in an atomic operation.
            temp_file = str(self.qas_data_results) + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(
                    flatten_lists_json_format(data, indent=4)
                )
            os.replace(temp_file, self.qas_data_results)
            self.last_save = datetime.now().timestamp()
            self.new_data = False
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            print(e)
        
        return self.qas_data_results
    
    def extract_doc_pages_as_images(self, resolution=144):
        assert self.documents_dir.exists(), 'unknown documents directory'
        print(f"\nExtracting document pages screenshots in {self.extracted_dir}\nNote: already extracted screenshots are ignored\n")

        uniq_pdf_names = list(dict.fromkeys(
            sample['doc_id'] for sample in self.load_data()
        ))

        for name in tqdm(uniq_pdf_names):
            doc_name = Path(name).stem

            with pymupdf.open(Path(self.documents_dir).joinpath(name)) as pdf:
                for page_idx, page in enumerate(pdf):
                    img_file = Path(self.BUILD_IMG_PATH(doc_name, page_idx))
                    img_file.parent.mkdir(exist_ok=True, parents=True)

                    if not img_file.exists():
                        img = page.get_pixmap(dpi=resolution)
                        img.save(img_file)

    def load_sample_images(self, sample: dict|str, img_indices: list[int] | None = None) -> list:
        assert self.extracted_dir.exists(), 'no extracted data found. You should probably extract first.'

        doc_name = sample['doc_id'] if isinstance(sample, dict) else sample
        doc_name = Path(doc_name).stem

        img_indices = img_indices if isinstance(img_indices, list) else itertools.count()

        images = []
        for idx in img_indices:
            img_file = self.BUILD_IMG_PATH(doc_name, idx)
            if not Path(img_file).exists():
                break
            images.append(Image.open(img_file))

        return images
        
def flatten_lists_json_format(data, indent=4):
    """
    Flatten nested results in the output JSON for improved readability.
    """
    def custom_json_dumps(obj, indent=4, level=0):
        space = ' ' * (indent * level)
        if isinstance(obj, list):
            return '[' + ', '.join(json.dumps(el, ensure_ascii=False) for el in obj) + ']'
        if isinstance(obj, dict) and level >= 2 and any(key in obj for key in ['answer', 'total', 'max_1', '1000']):
            return '{' + ', '.join(f'{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}' for k, v in obj.items()) + '}'
        elif isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                key_str = json.dumps(k, ensure_ascii=False)
                val_str = custom_json_dumps(v, indent, level + 1)
                items.append(f'{space}{" " * indent}{key_str}: {val_str}')

            return '{\n' + ',\n'.join(items) + f'\n{space}' + '}'
        else:
            return json.dumps(obj, ensure_ascii=False)  
          
    if isinstance(data, list):
        items = [custom_json_dumps(item, indent=indent, level=1) for item in data]
        return '[\n' + ',\n'.join(items) + '\n]'
    
    return custom_json_dumps(data, indent=indent, level=0)