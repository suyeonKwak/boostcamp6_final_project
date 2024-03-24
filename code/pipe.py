import torch

# from diffusers import LCMScheduler, AutoPipelineForText2Image

from diffusers import DiffusionPipeline
from deep_translator import GoogleTranslator

from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use


# nltk.download("punkt")


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
def summarize_text(text: str, num_beams: int = 8):

    model, tokenizer = build_text_model()

    max_input_length = 512 + 256

    inputs = ["summarize: " + text]

    inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
    )
    output = model.generate(
        **inputs, num_beams=num_beams, do_sample=True, min_length=5, max_length=15
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


def draw_image(
    summary: str,
    mode: str = "summary",
    num_inference_steps: int = 20,
    guidance_scale: int = 8.5,
):

    pipe = build_t2i_model()

    if mode == "summary":
        # korean to English
        text = GoogleTranslator(source="ko", target="en").translate(summary)

        prompt = "A drawing of the scene that " + text  # + " in the style of Monet"
        print(f"prompt : {prompt}")

    elif mode == "keyword":
        prompt = "A painting with " + ", ".join(summary[:-1]) + ", and " + summary[-1]
        print(f"prompt : {prompt}")

    # disable guidance_scale by passing 0
    generated_image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    title = GoogleTranslator(source="en", target="ko").translate(prompt)

    return generated_image, title


if __name__ == "__main__":

    test_texts = [
        """
        오늘은 온도가 20도까지 올라갔다. 만연한 봄이 왔다. 날씨가 좋아서 산책을 했다. /
        꽃도 보고 카페에 들려 커피도 마셨다. 바쁜 평일을 보내고 이렇게 주말을 여유롭게 /
        보내니 소소한 행복을 누리는 것 같아 좋았다.""",
        """
        월요일이다.. 아침에 비가 왔다. 안그래도 월요일이라 우울했는데 비까지 와서 더 우울했다./
        주말까지 언제 기다리지? 오늘은 학생식당에서 밥을 먹었다. 동기들이랑 연구실에서 각자 공부하다/
        시간 되면 밥 먹으러가고 다시 연구하다 칼퇴하는 게 일상이 됐다. 오늘 하루도 고생했다~""",
        """
        기다리고 기다리던 아이유 콘서트를 보고 왔다!! 아이유가 아이유했다. 노래 "Love wins all"을 /
        들을 때는 어찌나 울컥하던지 나도 모르게 눈물이 났다. 콘서트장을 가득 채우던 보랏빛 물결은 /
        너무나도 아름다웠다. 다음 콘서트에서도 내 자리 있었으면!! ㅠㅠ 콘서트 다녀오니 체력이 바닥 났다. /
        집 가자마자 드러누웠다. """,
        """
        친구를 만났다. 5년만에 만난 고등학교 친군데 오랜만에 만나서 그런가 더 반가웠다ㅠㅠ /
        변함없이 한결같은 모습에 마치 고등학교 때로 돌아간것만 같았다. 그동안 어떻게 지냈는지, /
        지금은 뭐하고 있는지 등 근황 얘기를 하다보니 시간이 훅 갔다. 종종 이렇게 연락하고 지내면 좋겠다.
        """,
        """
        특별한 일 없이 잔잔히 지나간 하루였다. 살짝 쌀쌀해서 목도리를 챙겨갔다. /
        저녁에 심야영화를 봤다. 듄2를 봤는데 기대 이상이었다. 그 웅장함은 /
        시즌 1보다 더 업그레이드 되서 돌아온 것 같았다. 남자친구도 듄을 보면 참 좋을텐데/
        시즌 3 같이 보러가고 싶은데 너무 아쉽다.ㅠㅠ 시즌 3은 사랑과 질투 얘기라니/
        더욱 재밌을 것 같다. ㅎㅎ""",
        """
        5개월간의 대장정! 드디어 끝이 보인다! 부스트캠프 AI Tech 6기 하는 동안 /
        정말 많이 배우고 많이 성장한 것 같다. 100프로 흡수 했다면 더더욱 성장했을테지만, /
        지금도 만족한다. 덕분에 혼자서 개인 프로젝트도 뚝딱 할 수 있을 정도로 알찬 내용들을 /
        배울 수 있었다. 만약 후배 중에서 AI 분야로 취업하고 싶은데 부스트캠프를 추천하냐고 물어보면 /
        당연히 YES다!! 더군다나 좋은 팀원들을 만나 서로 동기부여하고 부족한 부분을 매꿔줄 수 있어 정말 /
        운이 좋았다고 생각한다. 앞으로도 자주 연락하는 사이로 남고 싶다. 마지막에 끝까지 같이 /
        못한게 많이 아쉽지만 그래도 잊지않고 불러주고 공유해줘서 정말 고마웠다. 모두들 좋은 /
        결과가 있었으면 좋겠다!
        """,
        """
        바다 드라이브를 하고 왔다. 작년부터 동해에 가보고 싶다는 생각을 하고 있었는데, 좀처럼 /
        시간이 나지 않아서 미루고 있다가 드디어 다녀왔다. 푸른 하늘 아래 푸른 바다는 정말 아름다웠다. /
        바다를 바라보며 멍때리는데 생각도 정리되고 평화롭고 정말 좋았다. 점심은 근처에서 칼국수를 먹었다. /
        그냥 우연히 지나가다 들어간 집이었는데 숨은 맛집이었다! 국물이 칼칼한 게 아주 좋았다. /
        이후에 근처 카페에서 차한잔 하고 좀 쉬었다가 다시 집으로 돌아왔다. 빨리 운전 면허 따서 /
        혼자도 바다 드라이브 와야지~
        """,
        """
        눈이 왔다! 스키장 가는 날 눈이 오다니 운이 좋았다. 리프트 타고 올라가는데 풍경이 너무 좋았다. /
        무릎을 다쳐서 스키다고 내려오는 게 조금 겁났다. 근데 몇번 타고 나니 두려움은 잊어버렸고 /
        마냥 즐겁고 재밌었다. 스키는 참 하체 운동이 잘 되는 것 같다. 집에 오니 근육통이 장난 아니다.. /
        내일도 주말이니까 늦잠 자야지ㅎㅎ
        """,
        """
        집에서 뒹굴뒹굴 거렸다. 넷플릭스에 재밌는 드라마가 얼마나 많이 올라왔는지, 하루종일 밀린 드라마를 /
        봤다. 요즘 계속 앉거나 누워있기만 해서 그런가 살이 많이 쪘다ㅠㅠ 언제 빼지.. 남자친구는 옆에서 뱃살 나왔다고 /
        어찌나 놀리는지 얄미워 죽겠다. 내가 어! 여름 전에 살빼고 말지!!
        """,
    ]
    # text = """
    #     오늘은 온도가 20도까지 올라갔다. 만연한 봄이 왔다. 날씨가 좋아서 산책을 했다. /
    #     꽃도 보고 카페에 들려 커피도 마셨다. 바쁜 평일을 보내고 이렇게 주말을 여유롭게 /
    #     보내니 소소한 행복을 누리는 것 같아 좋았다."""

    mode = "summary"  # keyword
    num_inference_steps = 30
    guidance_scale = 8.5
    grid = {
        "num_inference_steps": [10, 20, 30, 40, 50],
        "guidance_scale": [5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12],
    }

    for gs in grid["guidance_scale"]:
        for nis in grid["num_inference_steps"]:
            print(f"num_inference_steps : {nis} \nguidance_scale : {gs} \n")
            images = []
            titles = []
            for text in test_texts:
                if mode == "summary":
                    summary = summarize_text(text)
                elif mode == "keyword":
                    summary = keyword_extract(text)

                generated_image, title = draw_image(summary, mode=mode)

                images.append(generated_image)
                titles.append(title)

            # 이미지의 가로, 세로 크기 추출
            widths, heights = zip(*(i.size for i in images))

            # 새로 생성될 이미지의 크기 계산
            total_width = max(widths) * 3
            max_height = max(heights) * 3

            # 새로운 이미지 생성 (배경 투명)
            new_image = Image.new("RGBA", (total_width, max_height))

            # 이미지를 이어붙이기
            x_offset = 0
            y_offset = 0
            for img in images:
                new_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                if x_offset >= total_width:
                    x_offset = 0
                    y_offset += img.height

            save_dir = "./code/test_images"
            file_count = len(
                [
                    name
                    for name in os.listdir(save_dir)
                    if os.path.isfile(os.path.join(save_dir, name)) and mode in name
                ]
            )
            new_image_path = os.path.join(
                save_dir, f"image_draw_{nis}_{gs}_{file_count + 1}.png"
            )
            new_image = new_image.resize((768, 768))

            new_image.save(new_image_path)
            for title in titles:
                print(title)
            print("-" * 50)
