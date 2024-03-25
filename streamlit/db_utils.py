import pandas as pd
from PIL import Image
from typing import Optional


def save_info(
    diary: str,
    prompt: str,
    image: Image,
    date: str,
    weather: str,
    user_id: Optional[str] = None,
):
    df = pd.read_csv("../db_test.csv")
    print(date)

    if df[df["user_id"] == user_id].shape[0] == 0:
        id = user_id + "0"
    else:
        id = user_id + str(df[df["user_id"] == user_id].shape[0])

    image_path = "./streamlit/images/" + user_id + ".jpg"

    try:
        image.save(image_path)
        print(f"이미지 저장 성공: {image_path}")
    except Exception:
        print("이미지 저장 실패")
        return False

    new_row = {
        "id": id,
        "diary": diary,
        "prompt": prompt,
        "user_id": user_id,
        "image_path": image_path,
        "weather": weather,
        "date": date,
    }

    df.loc[df.shape[0]] = new_row
    df.to_csv("./streamlit/db_test.csv", index=False)

    print("정보를 저장했습니다!")
    return True


def load_info(user_id: str, page: str = "diary"):
    df = pd.read_csv("./streamlit/db_test.csv")

    df_id = df[df["user_id"] == user_id]

    if page == "diary":
        return df_id["date"].tolist()

    elif page == "memory":
        df_id.sort_values(by="date", inplace=True)
        diary = df_id["diary"].tolist()
        prompt = df_id["prompt"].tolist()
        date = df_id["date"].tolist()
        weather = df_id["weather"].to_list()
        image = []

        for row in df_id["image_path"].items():
            img = Image.open(row[1])
            image.append(img)

        return diary, prompt, image, date, weather
