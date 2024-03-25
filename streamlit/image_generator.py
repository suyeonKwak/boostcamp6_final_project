import torch
from diffusers import DiffusionPipeline
from deep_translator import GoogleTranslator

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use


def build_t2i_model():
    model_id = "SaiRaj03/Text_To_Image"

    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./huggingface"
    )

    pipe.to("cuda")

    return pipe


def draw_image(summary: str, num_inference_steps: int = 50, guidance_scale: int = 12):

    pipe = build_t2i_model()

    # korean to English
    text = GoogleTranslator(source="ko", target="en").translate(summary)

    prompt = "A cartoon scene that " + text

    # disable guidance_scale by passing 0
    generated_image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    title = GoogleTranslator(source="en", target="ko").translate(prompt)

    return generated_image, title
