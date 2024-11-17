import streamlit as st
import io
import os, subprocess

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

from datetime import datetime
from database.database import engine
from sqlalchemy import text

from common.chat import init_chat_config_form, deepinfra_models, deepinfra_client
from common.chat import get_chat_client

from common.voice import init_voice_config_form, text_to_voice, voice_to_text

from utils import init_page_header, init_session_state, local_models, get_avatar
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited, check_use_limit
from utils import global_system_prompt, main_container_height


title = "语音合成"
icon = "🔊"
init_page_header(title, icon)
init_session_state()

localdir = f"users/{st.session_state.username}/records"


def select_voice_freq():
    count = 0
    try:
        with engine.connect() as conn:
            sql = text(f'''
            select count(*) from ai_labs_voice where username = :username and date_time >= current_date;
            ''')
            count = conn.execute(sql, [{'username': st.session_state.username}]).fetchone()[0]
    except Exception as e:
        st.exception(e)
    return count


def select_aigc_freq():
    st.session_state.aigc_temp_freq, st.session_state.aigc_perm_freq = select_aigc_left_freq()
    st.session_state.aigc_temp_voice = select_voice_freq()


def insert_voice(user_voice, user_text, assistant_voice, assistant_text):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_voice(username, user_voice, user_text, assistant_voice, assistant_text, date_time) 
            values(:username, :user_voice, :user_text,:assistant_voice,  :assistant_text, :date_time)
            ''')
            conn.execute(sql, [{
                'username': st.session_state.username,
                'user_voice': user_voice,
                'user_text': user_text,
                'assistant_voice': assistant_voice,
                'assistant_text': assistant_text,
                'date_time': date_time
            }])
            conn.commit()
    except Exception as e:
        st.exception(e)


def select_voice():
    with engine.connect() as conn:
        sql = text("""
            select user_voice, user_text, assistant_voice, assistant_text from (select * from ai_labs_voice where username = :username order by id desc limit :showlimit) temp order by id
        """)
        return conn.execute(sql, [{
            'username': st.session_state.username,
            'showlimit': st.session_state.showlimit
        }]).fetchall()


def cache_voice(user_voice_file, user_input_text):
    user_input = user_input_text
    with main_container.chat_message("user"):
        with st.spinner("处理中，请稍等..."):
            if user_voice_file is not None:
                st.audio(user_voice_file, format="wav")
                user_input = voice_to_text(localdir, filename)
                if st.session_state.config_voice_show_text:
                    st.write(user_input)
            else:
                st.write(user_input)

    with main_container.chat_message("assistant", avatar=get_avatar("myshell/melotts")):
        if check_use_limit and st.session_state.aigc_temp_voice >= st.session_state.aigc_temp_freq and st.session_state.aigc_perm_freq < 1:
            use_limited()
        else:
            with st.spinner("处理中，请稍等..."):
                if st.session_state.config_voice_voice_type == "对话":
                    messages = [{
                            "role": "system",
                            "content": st.session_state.config_chat_system_prompt if "config_chat_system_prompt" in st.session_state.keys() else global_system_prompt
                        }]
                    for user_voice, user_text, assistant_voice, assistant_text in select_voice():
                        messages.append({
                            "role": "user",
                            "content": user_text
                        })
                        messages.append({
                            "role": "assistant",
                            "content": assistant_text
                        })
                    messages.append({
                        "role": "user",
                        "content": user_input
                    })
                    if st.session_state.config_chat_model in local_models:
                        response = get_chat_client().chat.completions.create(
                            model=st.session_state.config_chat_model,
                            messages=messages,
                            stream=False,
                            max_tokens=st.session_state.config_chat_max_tokens,
                            temperature=st.session_state.config_chat_temperature,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                            top_p=st.session_state.config_chat_top_p,
                        )
                        if response:
                            if hasattr(response.choices[0].message, "content"):
                                assistant_text = response.choices[0].message.content
                    elif st.session_state.config_chat_model in deepinfra_models:
                        response = deepinfra_client.chat.completions.create(
                            model=st.session_state.config_chat_model,
                            messages=messages,
                            stream=False,
                            max_tokens=st.session_state.config_chat_max_tokens,
                            temperature=st.session_state.config_chat_temperature,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                            top_p=st.session_state.config_chat_top_p,
                        )
                        if response:
                            if hasattr(response.choices[0].message, "content"):
                                assistant_text = response.choices[0].message.content
                else:
                    assistant_text = user_input

                assistant_voice = text_to_voice(assistant_text)
                st.audio(assistant_voice, format="wav")
                if st.session_state.config_voice_show_text:
                    st.write(assistant_text)
                insert_voice(user_voice_file, user_input, assistant_voice, assistant_text)
                if check_use_limit and st.session_state.aigc_temp_voice >= st.session_state.aigc_temp_freq:
                    update_aigc_perm_freq(-1)
                select_aigc_freq()


if __name__ == '__main__':

    select_aigc_freq()
    
    with st.sidebar:
        tabs = st.tabs(["语音设置", "模型设置"])
        with tabs[0]:
            init_voice_config_form()
            st.selectbox("合成类型", key="config_voice_voice_type", options=["复述", "对话"])
            st.toggle("显示文字", key="config_voice_show_text")
        with tabs[1]:
            init_chat_config_form()

        cols = st.columns(5)
        with cols[2]:
            audio_bytes = audio_recorder(text="", pause_threshold=2.5, icon_size='2x', sample_rate=16000)
        if check_use_limit:
            st.info(f"免费次数已用：{min(st.session_state.aigc_temp_freq, st.session_state.aigc_temp_voice)}/{st.session_state.aigc_temp_freq} 次。\n\r永久次数剩余：{st.session_state.aigc_perm_freq} 次。", icon="🙂")

    main_container = st.container(height=main_container_height)

    for user_voice, user_text, assistant_voice, assistant_text in select_voice():
        try:
            with main_container.chat_message("user"):
                if user_voice:
                    st.audio(user_voice, format="wav")
                    if st.session_state.config_voice_show_text:
                        st.write(user_text)
                else:
                    st.write(user_text)
            with main_container.chat_message("assistant", avatar=get_avatar("myshell/melotts")):
                st.audio(assistant_voice, format="wav")
                if st.session_state.config_voice_show_text:
                    st.write(assistant_text)
        except:
            pass

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        cache_voice(None, user_input_text)
    elif audio_bytes:
        os.makedirs(localdir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.wav"
        filepath = f"{localdir}/{filename}"

        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        audio_segment.export(filepath, format='wav')

        cache_voice(filepath, None)
