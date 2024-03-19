import pandas as pd
from PIL import Image


def save_info(diary: str, prompt: str, user_id: str):
    df = pd.read("./DB/db.csv")

    image_path = ""
    id = ""

    new_row = {
        "id": id,
        "diary": diary,
        "prompt": prompt,
        "user_id": user_id,
        "image_path": image_path,
    }

    df = df.append(new_row, ignore_index=True)

    print("정보를 저장했습니다!")
    return True


def load_info(user_id: str):
    df = pd.read("./DB/db.csv")

    df_id = df[df["user_id"] == user_id]

    diary = df_id["diary"].tolist()
    prompt = df_id["prompt"].tolist()
    image = []

    for row in df_id["image_path"].iterrows():
        img = Image.open(row[0])
        image.append(img)

    return diary, prompt, image
