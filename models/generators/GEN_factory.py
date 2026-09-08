
from .qwen import Qwen2_5_VL_7B_Instruct, Qwen3_VL_8B_Instruct
from .internvl import InternVL_3_5_8b_Instruct
from .llava import Llava_OneVision_Qwen2_7B
from .idefics import Idefics3_8B_Llama3

from models.generators import BaseGenerator

AVAILABLE_LVLM = {
    'Qwen2.5-VL-7B-Instruct': Qwen2_5_VL_7B_Instruct,
    'Qwen3-VL-8B-Instruct': Qwen3_VL_8B_Instruct,
    'InternVL3_5-8B-Instruct': InternVL_3_5_8b_Instruct,
    'Idefics3-8B-Llama3': Idefics3_8B_Llama3,
    'llava-onevision-qwen2-7b': Llava_OneVision_Qwen2_7B,
}

def get_generator(name: str) -> BaseGenerator:
    try:
        return AVAILABLE_LVLM[name]()
    except KeyError:
        raise ValueError(f"Unknown generator: {name}")
