import streamlit as st
import os, subprocess

import requests, json
import uuid
from datetime import datetime

from utils import get_config


def init_voice_config_form():
    model = st.selectbox("Model", key="config_voice_model", options=["melotts"])
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", key="config_voice_temperature", min_value=0.1, max_value=2.0, value=0.3, step=0.1)
        top_p = st.slider("Top P", key="config_voice_top_p", min_value=1.0, max_value=5.0, value=0.7, step=1.0)
    with col2:
        seed = st.slider("Seed", key="config_voice_seed", min_value=1, max_value=1000, value=42, step=1)
        top_k = st.slider("Top K", key="config_voice_top_k", min_value=1, max_value=100, value=20, step=1)


def text_to_voice(voice_text):
    localdir = f"users/{st.session_state.username}/voices"
    os.makedirs(localdir, exist_ok=True)
    localfile = f"{localdir}/{uuid.uuid4()}.wav"

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    data=json.dumps({
                         'text': voice_text,
                         'temperature': st.session_state.config_voice_temperature if "config_voice_temperature" in st.session_state.keys() else 0.3,
                         'top_P': st.session_state.config_voice_top_p if "config_voice_top_p" in st.session_state.keys() else 0.7,
                         'top_K': st.session_state.config_voice_top_k if "config_voice_top_k" in st.session_state.keys() else 20,
                         'audio_seed_input': st.session_state.config_voice_seed if "config_voice_seed" in st.session_state.keys() else 42,
                         'text_seed_input': st.session_state.config_voice_seed if "config_voice_seed" in st.session_state.keys() else 42,
                    })
    response = requests.post(url=f"http://{get_config('server_address')}/tts/{st.session_state.config_voice_model if 'config_voice_model' in st.session_state.keys() else 'melotts'}/generate_audio_file", headers=headers, data=data)

    with open(localfile, "wb") as file:
        file.write(response.content)

    return localfile

def voice_to_text(localdir, filename):
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
    files = {
        "audio": (filename, open(f"{localdir}/{filename}", 'rb'))
    }
    response = requests.post(url=f"http://{get_config('server_address')}/asr/funasr/audio_to_text", headers=headers, files=files)

    return response.text