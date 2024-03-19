import streamlit as st
from streamlit_option_menu import option_menu
import time
from datetime import datetime
from image_generator import draw_image
from summary_generator import summarize_text

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
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
    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("오늘의 날짜", value=datetime.now())

    with col2:
        weather = st.selectbox(
            "오늘의 날씨",
            ("맑음☀️", "흐림☁️", "눈☃️", "비☔"),
            index=None,
            placeholder="선택해주세요!",
        )
        year, month, day = date.year, date.month, date.day

    msg = st.text_area("일기장", height=10)
    # st.subheader(msg)

    # user_input = st.chat_input(f"오늘은 {year}년 {month}월 {day}일")

    if msg:
        if st.button("작성 완료"):
            with st.spinner("그림을 생성하고 있어요"):
                time.sleep(10)
                # 요약문 생상
                summary = summarize_text(msg)
                print("summary:", summary)

                # prompt -> 이미지 생성
                generated_image, title = draw_image(summary)

            st.write(f"{year}년 {month}월 {day}일 | 날씨 : {weather}")
            st.write(msg)
            # st.image("./냥이.jpg")
            st.image(generated_image)
            st.caption(title)

        # 정보 저장


elif choose == "Memory":
    st.write("gallery")

elif choose == "About":
    st.write(
        """
    당신의 이야기를 그림과 함께 남겨보아요! \n
    AI를 활용한 그림일기 데모 페이지 입니다.
    """
    )
