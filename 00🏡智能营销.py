import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.app_logo import add_logo

from datetime import datetime
from database.database import engine
from sqlalchemy import text

import json
import hashlib

from utils import init_page_header, init_session_state, load_lottiefile, get_avatar

title = "智能营销助手"
icon = "🏡"
st.set_page_config(
    page_title=title,
    page_icon=icon,
    layout="wide",
    menu_items={
        "Get Help": "https://ai.gitee.com/wux-labs/smart-sales",
        "Report a bug": "https://ai.gitee.com/wux-labs/smart-sales/issues",
        "About": """
## 🏡智能营销助手

众所周知，获客、活客、留客是电商行业的三大难题，谁拥有跟客户最佳的沟通方式，谁就拥有客户。

随着用户消费逐渐转移至线上，电商行业面临以下一些问题：

* 用户交流体验差
* 商品推荐不精准
* 客户转化率低
* 退换货频率高
* 物流成本高

在这样的背景下，未来销售的引擎——大模型加持的智能营销助手就诞生了。

它能够与用户的对话，了解用户的需求，基于多模态的AIGC生成能力，持续输出更符合用户消费习惯的文本、图片和视频等营销内容，推荐符合用户的商品，将营销与经营结合。

""",
    }
)
init_session_state()
# add_logo("statics/doc/logo.png", height=160)
st_lottie(load_lottiefile("statics/image/welcome.json"), speed=1, reverse=False, loop=True, quality="high", height=200)
st.subheader(f"{icon}{title}", divider='rainbow')


def check_username(username):
    # try:
    count = 0
    with engine.connect() as conn:
        sql = text(f'''
        select count(*) from ai_labs_user where username = :username;
        ''')
        count = conn.execute(sql, [{'username': username}]).fetchone()[0]
    return count
    # except:
    #   pass


def check_login(username, userpass):
    # try:
    count = 0
    with engine.connect() as conn:
        sql = text(f'''
        select count(*) from ai_labs_user where username = :username and userpass = :userpass;
        ''')
        count = conn.execute(sql, [{'username': username, 'userpass': userpass}]).fetchone()[0]
    return count
    # except:
    #   pass


def insert_user(username, fullname, rolename, gender, mailaddr, userpass):
    # try:
    with engine.connect() as conn:
        date_time = datetime.now()
        sql = text(f'''
        insert into ai_labs_user(username, fullname, rolename, gender, mailaddr, userpass, aigc_temp_freq, aigc_perm_freq, date_time)
        values(:username, :fullname, :rolename, :gender, :mailaddr, :userpass, 3, 0, :date_time)
        ''')
        count = conn.execute(sql, [{
            'username': username,
            'fullname': fullname,
            'rolename': rolename,
            'gender': gender,
            'mailaddr': mailaddr,
            'userpass': userpass,
            'date_time': date_time
        }])
        conn.commit()
    # except:
    #   pass


def select_user(username, userpass):
    # try:
    with engine.connect() as conn:
        date_time = datetime.now()
        sql = text(f'''
        select * from ai_labs_user where username = :username and userpass = :userpass;
        ''')
        return conn.execute(sql, [{'username': username, 'userpass': userpass}]).fetchone()
    # except:
    #   pass

with st.sidebar:
    if st.session_state.username == "guest":
        st.warning(f"""
        当前身份：{st.session_state["fullname"]}

        注册登录后可查看自己的数据！
        """)

        with st.expander("注册/登录"):
            tabs = st.tabs(["注册", "登录"])

            with tabs[0]:
                username = st.text_input("账号：", key="register_username")
                fullname = st.text_input("姓名：", key="register_fullname")
                rolename = st.selectbox("角色：", key="register_rolename", options=["买家", "卖家"])
                gender = st.selectbox("性别：", key="register_gender", options=["女", "男"])
                mailaddr = st.text_input("邮箱：", key="register_mailaddr")
                password1 = st.text_input("密码：", type="password", key="register_password1")
                password2 = st.text_input("确认：", type="password", key="register_password2")

                if st.button("注册", type="primary", use_container_width=True):
                    if not username or not fullname or not mailaddr or not password1:
                        st.error(f"请输入完整的注册信息！")
                    elif password1 != password2:
                        st.error(f"两次输入的密码不相同！")
                    elif check_username(username) > 0:
                        st.error(f"用户{username}已存在！")
                    else:
                        password = hashlib.md5(password1.encode('utf-8')).hexdigest()
                        # insert_user(username, fullname, rolename, gender, mailaddr, password)
                        st.info("注册成功，请登录！")

            with tabs[1]:
                username = st.text_input("账号：", key="login_username")
                password1 = st.text_input("密码：", type="password", key="login_password")
                if st.button("登录", type="primary", use_container_width=True):
                    password = hashlib.md5(password1.encode('utf-8')).hexdigest()
                    if check_login(username, password) == 0:
                        st.error(f"账号或密码错误！")
                    else:
                        user = select_user(username, password)
                        st.session_state["userid"] = user[0]
                        st.session_state["username"] = user[1]
                        st.session_state["fullname"] = user[2]
                        st.session_state["rolename"] = user[3]
                        st.session_state["gender"] = user[4]
                        st.session_state["mailaddr"] = user[5]
                        st.session_state["aigc_temp_freq"] = user[7]
                        st.info("登录成功！")
                        st.rerun()

    else:
        st.success(f"""
        欢迎您，{st.session_state["fullname"]}！

        希望您有一个愉快使用体验！
        """)

        if st.button("退出", key="logout-button", help="使用结束，退出当前登录啦~", type="primary",
                        use_container_width=True):
            del st.session_state["userid"]
            del st.session_state["username"]
            del st.session_state["fullname"]
            del st.session_state["rolename"]
            del st.session_state["gender"]
            del st.session_state["mailaddr"]
            del st.session_state["aigc_temp_freq"]
            init_session_state()
            st.rerun()

st.markdown("众所周知，获客、活客、留客是电商行业的三大难题，谁拥有跟客户最佳的沟通方式，谁就拥有客户。", unsafe_allow_html=True)

st.image("statics/doc/doc_image_01.png")
st.image("statics/doc/doc_image_02.png")
st.image("statics/doc/doc_image_03.png")
st.image("statics/doc/doc_image_04.png")
st.image("statics/doc/doc_image_05.png")
st.image("statics/doc/doc_image_06.png")
