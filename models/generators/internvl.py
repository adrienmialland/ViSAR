
import torch
from PIL.Image import Image
import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode
from transformers.utils import is_flash_attn_2_available
from transformers import AutoTokenizer, AutoModel

from models.generators import BaseGenerator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Implementation of InternVL3_5-8B-Instruct model using 
# the publicly available Hugging Face APIs. 
#
# see: https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct
class InternVL_3_5_8b_Instruct(BaseGenerator):
    def __init__(self):
        super().__init__()
        
        self.model: AutoModel = None
        self.tokenizer: AutoTokenizer = None

        self.model_name = None

    def from_pretrained(self):
        if self.model == None:
            use_flash_attn = is_flash_attn_2_available()

            self.model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL3_5-8B-Instruct",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=use_flash_attn,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                "OpenGVLab/InternVL3_5-8B-Instruct", trust_remote_code=True, use_fast=False
            )

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image: Image, input_size=448, max_num=12):
        # image = Image.open(image_file).convert('RGB')
        image = image.convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @torch.no_grad()
    def generate(self, query: str, images: list[Image]):
        generation_config = dict(
            max_new_tokens=2048, 
            do_sample=False,
            eos_token_id=151645, 
            pad_token_id=151645
        )

        pixel_values = [self.load_image(img, max_num=12).to(torch.bfloat16).cuda() for img in images]
        num_patches = [pix.size(0) for pix in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)

        query = "\n".join(
            [f"Image-{i+1}: <image>" for i in range(len(images))] + [query]
        )

        answer = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            query, 
            generation_config, 
            num_patches_list=num_patches, 
            history=None, 
            return_history=False
        )

        return answer


