

import json
import torch
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, PreTrainedModel
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from models.generators import BaseGenerator

# Aliases:
QwenProcessor = Qwen2_5_VLProcessor

SYSTEM_PROMPT = """You are a Visual Question Answering assistant. Your task is to use ONLY the provided document page images to answer the user's question.

Carefully analyze each image. Do NOT use external knowledge, prior assumptions, or hallucinations.

You should return ONLY the final answer in the following JSON format, with NO extra explanation:
{"Answer": <Your final answer here>}"""

USER_PROMPT = """Here are the user's question and the relevant document pages. Use only these images to answer the question.

Question: {question}"""

# Common implementation shared by Qwen models using the publicly
# available Hugging Face APIs. See the individual model classes
# below for model-specific behavior and the corresponding
# Hugging Face model links.
class QwenBaseModel(BaseGenerator):
    def __init__(self):
        super().__init__()
        
        self.model: PreTrainedModel = None
        self.processor: QwenProcessor = None

        self.parser = OutputParser()

        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT
        self.has_vision = False

        self.params = {'max_new_tokens': 2048}
    
    def from_pretrained(self):
        "Defined Below"

    def build_messages(self, query, images):
        user_prompt = self.user_prompt.replace("{question}", query)

        if isinstance(self, Qwen3_VL_8B_Instruct):
            sys_prompt = [{"type": "text", "text": self.system_prompt}]
        else:
            sys_prompt = self.system_prompt

        system_prompt_msg = [
            {"role": "system", "content": sys_prompt}
        ] if sys_prompt else []

        if self.has_vision:
            return system_prompt_msg + [
                {"role": "user", "content": 
                    [{"type": "text", "text": user_prompt}] +
                    [{"type": "image", "image": img} for img in images]
                }
            ]
        else:           
            return system_prompt_msg + [
                {"role": "user", "content": user_prompt}
            ]

    def encode_inputs(self, messages):
        if isinstance(self, Qwen3_VL_8B_Instruct):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )

            return prompt.to(self.model.device)
            
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.has_vision:
            images, videos = process_vision_info(messages)
            inputs = self.processor(
                text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[prompt], return_tensors="pt"
            )

        return inputs.to("cuda")
    
    def decode_outputs_tokens(self, output_tokens):
        output_txt = self.processor.batch_decode(
            output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_txt[0]
    
    def extract_output_tokens(self, inputs, outputs):
        return [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]

    @torch.no_grad()
    def generate(self, query: str, images: list[Image], no_parse=False):
        messages = self.build_messages(query, images)
        encoded_msg = self.encode_inputs(messages)

        output_tokens = self.model.generate(
            **encoded_msg, **self.params
        )

        answer_token = self.extract_output_tokens(encoded_msg, output_tokens)
        answer_texts = self.decode_outputs_tokens(answer_token)

        if no_parse:
            return answer_texts
        return self.parser.safe_parse_output(answer_texts)

# see: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
class Qwen2_5_7B_Instruct(QwenBaseModel):
    def __init__(self):
        super().__init__()
        self.has_vision = False

    def from_pretrained(self):
        if self.model == None:
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            dtype = "auto" if attn_impl == "eager" else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", dtype=dtype, device_map="auto", attn_implementation=attn_impl
            )
            self.processor = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=False)

# see: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
class Qwen2_5_14B_Instruct(QwenBaseModel):
    def __init__(self):
        super().__init__()
        self.has_vision = False

    def from_pretrained(self):
        if self.model == None:
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            dtype = "auto" if attn_impl == "eager" else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-14B-Instruct", dtype=dtype, device_map="auto", attn_implementation=attn_impl
            )
            self.processor = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", use_fast=False)  

# see: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
class Qwen2_5_VL_7B_Instruct(QwenBaseModel):
    def __init__(self):
        super().__init__()
        self.has_vision = True
        self.params['do_sample'] = False
        self.params['temperature'] = None

    def from_pretrained(self):
        if self.model == None:
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            dtype = "auto" if attn_impl == "eager" else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", dtype=dtype, device_map="auto", attn_implementation=attn_impl
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False)

# see: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
class Qwen3_VL_8B_Instruct(QwenBaseModel):
    def __init__(self):
        super().__init__()
        self.has_vision = True

    def from_pretrained(self):
        if self.model == None:
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            dtype = "auto" if attn_impl == "eager" else torch.bfloat16

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct", dtype=dtype, device_map="auto", attn_implementation=attn_impl
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")


class OutputParser():
    def safe_parse_output(self, model_output: str) -> str:
        model_output = model_output.strip()
    
        if model_output.lower() == 'none' or model_output is None:
            print('returned None', model_output)
            return model_output
        
        json_data = None

        try:
            json_data: dict[str, str] = json.loads(model_output)
        except json.JSONDecodeError:
            json_match = self.extract_json_from_output(model_output)
            if json_match:
                try:
                    json_data = json.loads(json_match)
                except json.JSONDecodeError:
                    pass
        
        if isinstance(json_data, dict):
            for k, v in json_data.items():
                if k.lower() == "answer":
                    return str(v).strip()
        elif isinstance(json_data, list):
            return str(json_data)
        
        print('could not parse model ouput:', model_output)
        return model_output
    
    def extract_json_from_output(self, output: str):
        output = output.strip()
        start = output.find('{')
        if start == -1:
            return None

        braces = 0
        for i, c in enumerate(output[start:], start=start):
            if c == '{':
                braces += 1
            elif c == '}':
                braces -= 1
                if braces == 0:
                    out = output[start:i+1]
                    while '{\n' in out or '\n}' in out:
                        out = out.replace('{\n', '{').replace('\n}', '}')
                    return out.replace("\n", "\\n")            
        return None