
import time
from tqdm import tqdm
from PIL.Image import Image

from data import Dataset
from config import GenConfig

from pipeline import RetrievalState
from models.generators import BaseGenerator, get_generator

TOTAL_PIXEL_BUDGET = 55_000_000

# Given the TOTAL_PIXEL_BUDGET, the images where actually never resized, since our experiments
# used LVLM input budget of Max-10 (see main text). Image resizing where actually only effective
# during preliminary experiments, where we used an input budget of Max-15.
def smart_resize_images(images: list[Image], total_pixel_budget:int=TOTAL_PIXEL_BUDGET, alpha:float=1.0):
    image_budget = alpha * total_pixel_budget / len(images)
    resized: list[Image] = []

    # 1. adjust images individually
    for img in images:
        H, W = img.size
        pixels = H * W

        if pixels <= image_budget:
            resized.append(img)
            continue
        
        scale = (image_budget / pixels) ** 0.5
        resized.append(img.resize((int(H * scale), int(W * scale))))

    total_pixels = sum(img.size[0] * img.size[1] for img in resized)

    # 2. adjust images globally
    if total_pixels > total_pixel_budget:
        scale = (total_pixel_budget / total_pixels) ** 0.5
        resized = [
            img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            for img in resized
        ]

    return resized

class AnswerGenerator():
    def __init__(self, gen: GenConfig):
        self.model: BaseGenerator = get_generator(gen.name)
        self.gen_id = gen.gen_id

    def is_answered(self, sample: dict[str, dict], state: RetrievalState):
        return (
            state.mode in sample 
            and state.encoder in sample[state.mode] 
            and self.gen_id in sample[state.mode][state.encoder]
        )

    def generate_answer(self, dataset: Dataset, state: RetrievalState):
        data = dataset.load_data(from_results=True)

        for sample in tqdm(data, desc='generation'):
            if self.is_answered(sample, state):
                continue           
            self.model.from_pretrained()

            k_star = sample[state.mode][state.encoder].get('k_star', None)
            indexs = sample[state.mode][state.encoder]['indexes'][:k_star]
            if k_star == 0:
                continue

            images = dataset.load_sample_images(sample, indexs[:state.max_k])
            images = smart_resize_images(images)

            start = time.perf_counter()
            answer = self.model.generate(sample['question'], images)
            latency = time.perf_counter() - start

            sample[state.mode][state.encoder].update({
                self.gen_id: {'answer': answer, 'latency': round(latency, 3)}
            })

            dataset.new_data = True
            dataset.save_results(data, 60)
        dataset.save_results(data)
