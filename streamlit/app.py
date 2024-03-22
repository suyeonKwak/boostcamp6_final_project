import streamlit as st
from streamlit_option_menu import option_menu
import time
from datetime import datetime
from image_generator import draw_image
from summary_generator import summarize_text
from db_utils import save_info, load_info

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


with st.sidebar:

    choose = option_menu(
        "Menu",
        ["Write Diary", "Memory", "About"],
        icons=["pencil", "book-half", "emoji-sunglasses"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#000000"},
            "icon": {"color": "#A3B899", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#CCCCCC",
            },
            "nav-link-selected": {"background-color": "#667B68"},
        },  # css 설정
    )

    # 유저 이름 받기 (간이 로그인)

    # 갤러리(마이페이지)
    # 유저id의 데이터 불러오기
    # 갤러리 형식으로 날짜와 그림 보여주기
    # 누르면 글도 보이게

    # 소개 링크
    "[Developed by Suyeon](https://github.com/suyeonKwak)"


st.title("🎨 Drawing Diary ")
st.caption("AI model serving by huggingface 🤗")


if choose == "Write Diary":

    col7, col8 = st.columns(2)
    col3, col4 = st.columns(2)

    with col7:
        user = st.text_input("Username", value="Anonymous")

        if user == "Anonymous":
            st.caption("익명으로 작성된 일기는 다시 볼 수 없어요")

    with col3:
        date = st.date_input("오늘의 날짜", value=datetime.now())
    if user != "Anonymous":
        dates = load_info(user_id=user, page="diary")
        # print(dates)
        if str(date) in dates:
            st.warning("해당 날짜에 이미 일기가 존재합니다.", icon="🚨")
        # print(date)

    with col4:
        weather = st.selectbox(
            "오늘의 날씨",
            ("맑음☀️", "흐림☁️", "눈☃️", "비☔"),
            index=None,
            placeholder="선택해주세요!",
        )
        year, month, day = date.year, date.month, date.day

    msg = st.text_area("일기장", height=300)
    # st.subheader(msg)

    # user_input = st.chat_input(f"오늘은 {year}년 {month}월 {day}일")

    if msg:
        if st.button("작성 끝"):
            with st.status("그림일기로 바꾸는 중이예요 ...", expanded=True) as status:
                st.write("작성한 일기를 분석중이예요 ...")
                # 요약문 생상
                summary = summarize_text(msg)
                print("summary:", summary)
                st.write("그림을 그리는 중이예요 ...")
                # prompt -> 이미지 생성
                generated_image, title = draw_image(summary)
                status.update(
                    label="그림일기 작성 끝!", state="complete", expanded=False
                )

            st.write(f"{year}년 {month}월 {day}일 | 날씨 : {weather}")
            st.write(msg)
            # st.image("./냥이.jpg")
            st.image(generated_image)
            st.caption(title)

            # 정보 저장
            save_info(
                diary=msg,
                prompt=title,
                user_id=user,
                image=generated_image,
                date=date,
                weather=weather,
            )


elif choose == "Memory":
    st.write("gallery")
    col1, col2 = st.columns(2)

    with col1:
        user = st.text_input("Username")

    if user == "":
        st.warning("Username을 입력해주세요", icon="⚠️")
        print("유저 입력 안함")
    elif user == "Anonymous":
        st.error("익명으로 작성된 일기는 볼 수 없어요", icon="🚫")
    else:
        diary, prompt, image, date, weather = load_info(user_id=user, page="memory")
        print("불러오기 성공")

        num = len(diary)
        if num == 0:
            st.write("작성된 일기가 없습니다😢")
        else:
            print(f"{user}의 저장된 일기 개수 : {num}")
            col5, col6 = st.columns(2)
            with col5:
                for i in range((num + 1) // 2):
                    idx = i * 2
                    globals()[f"expander{idx}"] = st.expander(
                        label=date[idx], expanded=False
                    )
                    globals()[f"expander{idx}"].write(
                        f"날씨 : {weather[idx]} \n\n" + diary[idx]
                    )
                    globals()[f"expander{idx}"].image(image[idx])
                    globals()[f"expander{idx}"].caption(prompt[idx])

            with col6:
                for i in range(num // 2):
                    idx = i * 2 + 1
                    globals()[f"expander{idx}"] = st.expander(
                        label=date[idx], expanded=False
                    )
                    globals()[f"expander{idx}"].write(
                        f"날씨 : {weather[idx]} \n\n" + diary[idx]
                    )
                    globals()[f"expander{idx}"].image(image[idx])
                    globals()[f"expander{idx}"].caption(prompt[idx])


elif choose == "About":
    st.write(
        """
    당신의 이야기를 그림과 함께 남겨보아요! \n
    AI를 활용한 그림일기 데모 페이지 입니다.
    """
    )
