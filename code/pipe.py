import torch

# from diffusers import LCMScheduler, AutoPipelineForText2Image

from diffusers import DiffusionPipeline
from deep_translator import GoogleTranslator

# from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use


nltk.download("punkt")


# build models #
def build_text_model():
    model_dir = "lcw99/t5-large-korean-text-summary"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="./huggingface")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, cache_dir="./huggingface")
    return model, tokenizer


def build_t2i_model():
    # model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    model_id = "SaiRaj03/Text_To_Image"
    # adapter_id = "latent-consistency/lcm-lora-sdxl"

    # pipe = AutoPipelineForText2Image.from_pretrained(
    #     model_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./huggingface"
    # )
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./huggingface"
    )
    # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # load and fuse lcm lora
    # pipe.load_lora_weights(adapter_id, cache_dir="./huggingface")
    # pipe.fuse_lora()

    return pipe


def build_keyword_extractor():
    # Load model directly
    model_dir = "transformer3/H2-keywordextractor"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="./huggingface")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, cache_dir="./huggingface")
    return tokenizer, model


# generator functions #
def summarize_text(text: str):

    model, tokenizer = build_text_model()

    max_input_length = 512 + 256

    inputs = ["summarize: " + text]

    inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
    )
    output = model.generate(
        **inputs, num_beams=8, do_sample=True, min_length=10, max_length=30
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    return predicted_title


def keyword_extract(text: str):

    tokenizer, model = build_keyword_extractor()
    en_text = GoogleTranslator(source="ko", target="en").translate(text)

    inputs = tokenizer(en_text, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    decoded_output = list(map(lambda x: x.rstrip().lstrip(), decoded_output.split(",")))
    print(f"\ndecoded_output : {decoded_output}")

    return decoded_output[:-1]


def draw_image(summary: str, mode: str = "summary"):

    pipe = build_t2i_model()

    if mode == "summary":
        # korean to English
        text = GoogleTranslator(source="ko", target="en").translate(summary)

        prompt = "A painting of the scene that " + text
        print(f"prompt : {prompt}")

    elif mode == "keyword":
        prompt = "A painting with " + ", ".join(summary[:-1]) + ", and " + summary[-1]
        print(f"prompt : {prompt}")

    # disable guidance_scale by passing 0
    generated_image = pipe(
        prompt=prompt, num_inference_steps=20, guidance_scale=8.5
    ).images[0]

    title = GoogleTranslator(source="en", target="ko").translate(prompt)

    return generated_image, title


if __name__ == "__main__":

    text = """
        오늘은 온도가 20도까지 올라갔다. 만연한 봄이 왔다. 날씨가 좋아서 산책을 했다. /
        꽃도 보고 카페에 들려 커피도 마셨다. 바쁜 평일을 보내고 이렇게 주말을 여유롭게 /
        보내니 소소한 행복을 누리는 것 같아 좋았다."""

    mode = "summary"  # keyword

    if mode == "summary":
        summary = summarize_text(text)
    elif mode == "keyword":
        summary = keyword_extract(text)

    generated_image, title = draw_image(summary, mode=mode)

    save_dir = "./code/test_images"
    file_count = len(
        [
            name
            for name in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, name)) and mode in name
        ]
    )
    new_image_path = os.path.join(save_dir, f"image_{mode}_{file_count + 1}.jpg")

    generated_image.save(new_image_path)
    print(title)
