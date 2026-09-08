
import torch
from PIL.Image import Image

from transformers import AutoProcessor, AutoModelForImageTextToText

from models.generators import BaseGenerator

SYSTEM_PROMPT = """
You are a Visual Question Answering assistant. Your task is to use ONLY the provided document page images to answer the user's question. 

Carefully analyze each image. Do NOT use external knowledge, prior assumptions, or hallucinations.

"""

# Implementation of Idefics3-8B-Llama3 model using 
# the publicly available Hugging Face APIs. 
#
# see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
class Idefics3_8B_Llama3(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = ''
    
    def from_pretrained(self):
        if self.model == None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                "HuggingFaceM4/Idefics3-8B-Llama3", dtype=torch.bfloat16
            ).to("cuda:0")
            self.processor = AutoProcessor.from_pretrained(
                "HuggingFaceM4/Idefics3-8B-Llama3"
            )

    def build_messages(self, query, num_images):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in range(num_images)
                ] + [
                    {"type": "text", "text": self.system_prompt + query},
                ]
            }
        ]

    @torch.no_grad()
    def generate(self, query: str, images: list[Image]):
        messages = self.build_messages(query, len(images))

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        
        )
        inputs = self.processor(
            text=prompt, 
            images=images, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs, max_new_tokens=2048
        )

        answer: str = self.processor.batch_decode(
            output, skip_special_tokens=True
        )[0]   

        answer = self.safe_parse_output(answer, query)

        return answer

    def safe_parse_output(self, model_output: str, query: str):
        assisant_idx = model_output.lower().find('assistant:')
        if assisant_idx != -1:
            return model_output[assisant_idx + len('assistant:'):].strip()
        
        query_idx = model_output.find(query)
        if query_idx != -1:
            answer_txt = model_output[query_idx + len(query):]

            if answer_txt.lower().strip().startswith('assistant'):
                answer_txt = answer_txt.strip()[len('assistant'):].strip()

            if answer_txt.startswith(':'):
                return answer_txt[1:].strip()

            return answer_txt
        
        prompt_idx = model_output.find(self.system_prompt)
        if prompt_idx != -1:
            return model_output[prompt_idx + len(prompt_idx):]

        sym_idx = model_output.rfind('<global-img>')
        if sym_idx != -1:
            return model_output[sym_idx + len(sym_idx):]

        print('could not parse model ouput:', model_output)
        return model_output

