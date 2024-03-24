import streamlit as st
from datetime import datetime
from DB.db_utils import load_info, save_info


def write_diary():
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
