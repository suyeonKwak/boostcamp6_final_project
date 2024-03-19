import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
from deep_translator import GoogleTranslator

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use


def build_t2i_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    return pipe


def draw_image(summary: str):

    pipe = build_t2i_model()

    # korean to English
    text = GoogleTranslator(source="ko", target="en").translate(summary)

    prompt = "A scene that " + text

    # disable guidance_scale by passing 0
    generated_image = pipe(
        prompt=prompt, num_inference_steps=20, guidance_scale=8.5
    ).images[0]

    title = GoogleTranslator(source="en", target="ko").translate(prompt)

    return generated_image, title
