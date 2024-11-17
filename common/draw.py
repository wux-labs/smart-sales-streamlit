import streamlit as st
import os
import json
import requests
import uuid
from datetime import datetime

from utils import get_config

def init_draw_config_form():
    model = st.selectbox("Model", key="config_image_model", options=["stable_diffusion_xl"])
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Width", key="config_image_width", min_value=1, max_value=2048, value=512, step=1)
        steps = st.slider("Steps", key="config_image_steps", min_value=1, max_value=100, value=20, step=1)
        sampler_name = st.selectbox("Sampler", key="config_image_sampler_name",
                                    options=["DDIM", "DPM++ 2M Karras", "DPM++ SDE Karras", "Heun"])
    with col2:
        height = st.slider("Height", key="config_image_height", min_value=1, max_value=2048, value=512, step=1)
        cfg_scale = st.slider("Scale", key="config_image_cfg_scale", min_value=1, max_value=32, value=7,
                            step=1)
        seed = st.number_input("Seed", key="config_image_seed", step=1, value=1)

    negative_prompt = st.text_area("Negative Prompt", key="config_image_negative_prompt", value="Disabled feet, abnormal feet, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW")


def save_draw_image(user_input_text):
    localdir = f"users/{st.session_state.username}/images"
    os.makedirs(localdir, exist_ok=True)
    localfile = f"{localdir}/{uuid.uuid4()}.png"

    params = {
        "prompt": user_input_text,
        "width": st.session_state.config_image_width if "config_image_width" in st.session_state.keys() else 1024,
        "height": st.session_state.config_image_height if "config_image_height" in st.session_state.keys() else 1024,
        "num_inference_steps": st.session_state.config_image_steps if "config_image_steps" in st.session_state.keys() else 20,
        "guidance_scale": st.session_state.config_image_cfg_scale if "config_image_cfg_scale" in st.session_state.keys() else 7,
        "seed": st.session_state.config_image_seed if "config_image_seed" in st.session_state.keys() else 1,
        "sampler_name": st.session_state.config_image_sampler_name if "config_image_sampler_name" in st.session_state.keys() else "DDIM",
        "negative_prompt": st.session_state.config_image_negative_prompt if "config_image_negative_prompt" in st.session_state.keys() else "Disabled feet, abnormal feet, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    data=json.dumps(params)
    response = requests.post(url=f"http://{get_config('server_address')}/image/{st.session_state.config_image_model}/generate_image_file", headers=headers, data=data)

    with open(localfile, "wb") as file:
        file.write(response.content)

    return localfile

