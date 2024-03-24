import sqlite3
import streamlit as st
from streamlit_option_menu import option_menu
from pages.diary import write_diary
from pages.memory import load_memory
from pages.about import about_me
from DB.db_utils import (
    check_hashes,
    create_diarytable,
    create_usertable,
    make_hashes,
    load_user_data,
    login_user,
    add_userdata,
    join_user,
)
import time


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "is_login" not in st.session_state or not st.session_state["is_login"]:
        return

    choose = authenticated_menu()
    if choose == "Write Diary":
        write_diary()
    elif choose == "Memory":
        load_memory()
    elif choose == "About":
        about_me()


def authenticated_menu():

    with st.sidebar:
        if st.sidebar.button("로그아웃"):
            st.session_state["is_login"] = False
            st.rerun()

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

    return choose


def login(user_c, diary_c, user_db, diary_db):

    with st.sidebar:
        st.header("로그인")
        username = st.text_input("ID")
        password = st.text_input("Password", type="password")

        login = st.button("로그인")
        signin = st.button("회원가입")

    if login:
        create_usertable(user_c)
        create_diarytable(diary_c)
        hashed_pswd = make_hashes(password)
        result = login_user(username, check_hashes(password, hashed_pswd), user_c)

        if result:
            st.session_state["is_login"] = True
            st.session_state["id"] = username
            st.session_state["my_data"] = load_user_data(username, diary_c)
            menu()
        else:
            st.sidebar.warning("아이디 혹은 비밀번호가 틀렸습니다.")

    if signin:
        create_usertable(user_c)
    if not password:
        st.sidebar.error("비밀번호를 입력해주세요")
        return
    result = join_user(username, user_c)
    if result:
        st.sidebar.error("이미 존재하는 아이디입니다.")
    else:
        add_userdata(username, make_hashes(password), user_c, user_db)
        user_db.commit()
        st.sidebar.success(f"가입을 환영합니다 {username}님")
        st.session_state["is_login"] = True
        st.session_state["id"] = username
        create_diarytable(diary_c)
        st.session_state["my_data"] = load_user_data(username, diary_c)
        time.sleep(2)
        menu()


if __name__ == "__main__":

    # DB 연결
    user_db = sqlite3.connect("./streamlit/DB/users_db.sqlite")
    diary_db = sqlite3.connect("./streamlit/DB/diary_db.sqlite")
    user_c = user_db.cursor()
    diary_c = diary_db.cursor()

    for key in st.session_state.keys():
        del st.session_state[key]

    st.session_state["is_login"] = False
    st.session_state["my_data"] = ""
    st.session_state["id"] = ""

    menu()
    login(user_c, diary_c, user_db, diary_db)
