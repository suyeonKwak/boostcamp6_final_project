import sqlite3
import streamlit as st
from streamlit_option_menu import option_menu
from pages.diary import write_diary
from pages.memory import load_memory
from pages.about import about_me
import hashlib
from DB.db_utils import *


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


def login():

    sidebar_title = st.sidebar.header('로그인')
    username = st.sidebar.text_input("ID")
    password = st.sidebar.text_input("Password",type='password')
    login = st.sidebar.button('로그인')
    signin = st.sidebar.button('회원가입')

    if login:
        create_usertable()
        create_diarytable()
        hashed_pswd = make_hashes(password)
        result = login_user(username,check_hashes(password,hashed_pswd))

    if result:
        st.session_state['is_login'] = True
        st.session_state['id'] = username
        st.session_state['my_data'] = load_user_data(username)
        st.session_state['today_data'] = st.session_state['my_data'][st.session_state['my_data']['date']==str(today)]
        menu()
    else:
        st.sidebar.warning("아이디 혹은 비밀번호가 틀렸습니다.")

    if signin:
    create_usertable(c=user_c) 
    if not password:
        st.sidebar.error('비밀번호를 입력해주세요')
        return
    result = join_user(username)
    if result:
        st.sidebar.error('이미 존재하는 아이디입니다.')
    else:
        add_userdata(username,make_hashes(password))
        user_db.commit()
        st.sidebar.success(f'가입을 환영합니다 {username}님')
        st.session_state['is_login'] = True
        st.session_state['id'] = username
        create_diarytable()
        st.session_state['my_data'] = load_user_data(username)
        st.session_state['today_data'] = st.session_state['my_data'][st.session_state['my_data']['date']==str(today)]
        time.sleep(2)
        st.switch_page('pages/diary.py')



if __name__ == "__main__":

    # DB 연결
    user_db = sqlite3.connect("./streamlit/DB/users.splite")
    diary_db = sqlite3.connect("./steramlit/DB/diarys.sqlite")
    user_c = user_db.cursor()
    diary_c = diary_db.cursor()

    for key in st.session_state.keys():
        del st.session_state[key]

    st.session_state["is_login"] = False
    st.session_state["my_data"] = ""
    st.session_state["id"] = ""

    menu()
    login()
