import streamlit as st


def load_memory():
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
