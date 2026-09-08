
import torch
from PIL.Image import Image

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import LlavaModel, LlavaProcessor

from models.generators import BaseGenerator

SYSTEM_PROMPT = """
You are a Visual Question Answering assistant. Your task is to use ONLY the provided document page images to answer the user's question. 

Carefully analyze each image. Do NOT use external knowledge, prior assumptions, or hallucinations.

"""

# Implementation of llava-onevision-qwen2-7b model using 
# the publicly available Hugging Face APIs. 
#
# see: https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf
class Llava_OneVision_Qwen2_7B(BaseGenerator):
    def __init__(self):
        super().__init__()
        
        self.model: LlavaModel = None
        self.processor: LlavaProcessor = None

        self.system_prompt = SYSTEM_PROMPT

    def from_pretrained(self):
        if self.model == None:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                "llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16, low_cpu_mem_usage=True              
            ).to(0)

            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-onevision-qwen2-7b-ov-hf", use_fast=True
            )
        
    def build_messages(self, query, num_images):
        return [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": self.system_prompt + query},
                ] + [
                    {"type": "image"} for _ in range(num_images)
                ]
            }
        ]

    def encode_inputs(self, messages):
        prompt = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )

        return prompt

    @torch.no_grad()
    def generate(self, query: str, images: list[Image]):
        torch.cuda.empty_cache()

        messages = self.build_messages(query, len(images))
        prompt = self.encode_inputs(messages)

        inputs = self.processor(
            images=images, 
            text=prompt, 
            return_tensors="pt"
        ).to(self.model.device)

        output_tokens = self.model.generate(
            **inputs, max_new_tokens=2048, pad_token_id=151645
        )

        text_answer: str = self.processor.decode(
            output_tokens[0][2:], skip_special_tokens=True, do_sample=False
        )

        return self.safe_parse_output(text_answer, query)

    def safe_parse_output(self, model_output: str, query: str):       
        answer_txt = model_output

        query_idx = answer_txt.lower().find(query.lower())
        if query_idx != -1:
            answer_txt = answer_txt[query_idx + len(query):]

        assisant_idx = answer_txt.lower().find('assistant')
        if assisant_idx != -1:
            return answer_txt[assisant_idx + len('assistant'):].strip()

        prompt_idx = model_output.lower().find(self.system_prompt.lower())
        if prompt_idx != -1:
            return model_output[prompt_idx + len(prompt_idx):].strip()

        print('could not parse model ouput:', model_output)
        return model_output
