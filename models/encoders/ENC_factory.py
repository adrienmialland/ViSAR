
from .vidore import Colpali_v1_2, ColQwen2_5_v0_1
from .modernvbert import ColModern_VBert

from models.encoders import BaseEncoder

AVAILABLE_ENCODER = {
    'colpali-v1.2': Colpali_v1_2,
    'colqwen2.5-v0.1': ColQwen2_5_v0_1,
    'colmodernvbert': ColModern_VBert
}

def get_encoder(name: str) -> BaseEncoder:
    try:
        encoder: BaseEncoder = AVAILABLE_ENCODER[name]()
        encoder.name = name
        return encoder
    except KeyError:
        raise ValueError(f"Unknown encoder: {name}")