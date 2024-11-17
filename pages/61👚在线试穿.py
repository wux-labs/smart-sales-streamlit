import streamlit as st
from streamlit_drawable_canvas import st_canvas

import os
import uuid
import numpy as np
import base64
from io import BytesIO
from datetime import datetime

from database.database import engine
from sqlalchemy import text

import json
import requests

import einops
import torch
from PIL import Image

from utils import init_page_header, init_session_state, get_config

title = "在线试穿"
icon = "👚"
init_page_header(title, icon)
init_session_state()

def query_produc_info(id):
    with engine.connect() as conn:
        sql = text("""
            select image from ai_labs_product_info where id = :id
        """)
        return conn.execute(sql, [{
            'id': id
        }]).fetchone()


if __name__ == '__main__':
    id = None
    refs_background_image = None
    base_background_image = None
    with st.sidebar:
        if "id" in st.query_params.keys():
            id = st.query_params["id"]
            refs_background_image = Image.open(query_produc_info(id)[0])
        elif "tryon_id" in st.session_state.keys():
            id = st.session_state.tryon_id
            refs_background_image = Image.open(query_produc_info(id)[0])
        tabs = st.tabs(["画笔设置", "图片设置", "高级设置"])
        with tabs[0]:
            stroke_color = st.color_picker("画笔颜色","#CCCCCC")
            stroke_width = st.slider("画笔宽度", 1, 100, 30)
        with tabs[1]:
            if id is None:
                refs_file_uploader = st.file_uploader("衣服平铺图", type=["png","jpg"])
                if refs_file_uploader:
                    refs_background_image = Image.open(refs_file_uploader)

            camera_flag = st.toggle("拍摄照片")
            if camera_flag:
                base_file_camera = st.camera_input("人体模特图")
                if base_file_camera:
                    base_background_image = Image.open(base_file_camera).resize((798, 600))
                    st.write(base_background_image.size)
            else:
                base_file_uploader = st.file_uploader("人体模特图", type=["png","jpg"])
                if base_file_uploader:
                    base_background_image = Image.open(base_file_uploader)
        with tabs[2]:
            model = st.selectbox("Model", key="config_tryon_model", options=["anydoor", "catvton"])
            strength = st.slider(label="Control Strength", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
            ddim_steps = st.slider(label="Steps", min_value=5, max_value=50, value=30, step=1)
            scale = st.slider(label="Guidance Scale", min_value=0.1, max_value=30.0, value=2.5, step=0.1)
            seed = st.slider(label="Seed", min_value=-1, max_value=99, step=1, value=42)
            # reference_mask_refine = st.checkbox(label='Reference Mask Refine', value=False)
            enable_shape_control = st.checkbox(label='Enable Shape Control', value=False)

    cols = st.columns(2)
    with cols[0]:
        st.write("衣服平铺图： 🚨请手动涂抹图片中需要换装的部分！")
        refs = st_canvas(key="canvas_refs",
                background_image=refs_background_image,
                height=798,
                width=600,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                )
    with cols[1]:
        st.write("人体模特图： 🚨请手动涂抹图片中需要换装的部分！")
        base = st_canvas(key="canvas_base",
                background_image=base_background_image,
                height=798,
                width=600,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                )
    cols = st.columns(3)
    with cols[1]:
        tryon1 = st.button("试穿", type="primary", use_container_width=True)
    # with cols[1]:
    #     tryon2 = st.button("试穿方式二", type="primary", use_container_width=True)

    if tryon1 and base_background_image and refs_background_image:
        base_image = base_background_image.convert("RGB")
        base_mask = Image.fromarray(base.image_data).resize(base_image.size).convert("L")

        refs_image = refs_background_image.resize(base_image.size).convert("RGB")
        refs_mask = Image.fromarray(refs.image_data).resize(base_image.size).convert("L")

        localdir = f"users/{st.session_state.username}/images/{uuid.uuid4()}"
        os.makedirs(localdir, exist_ok=True)

        localfile = f"{localdir}/tryon.png"

        clothfile = f"{localdir}/cloth.png"
        clothmask = f"{localdir}/cloth_mask.png"
        personfile = f"{localdir}/person.png"
        personmask = f"{localdir}/person_mask.png"

        base_image.save(personfile)
        base_mask.save(personmask)
        refs_image.save(clothfile)
        refs_mask.save(clothmask)

        headers = {
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        data = {
            "strength": strength,
            "num_inference_steps": ddim_steps,
            "guidance_scale": scale,
            "seed": seed,
            "enable_shape_control": enable_shape_control
        }
        files = {
            "person_image": ("person.png", open(personfile, 'rb')),
            "person_mask": ("person_mask.png", open(personmask, 'rb')),
            "cloth_image": ("cloth.png", open(clothfile, 'rb')),
            "cloth_mask": ("cloth_mask.png", open(clothmask, 'rb'))
        }
        response = requests.post(url=f"http://{get_config('server_address')}/try_on/{model}/tryon_file", headers=headers, data=data, files=files)

        with open(localfile, "wb") as file:
            file.write(response.content)

        st.write("试穿效果图")
        st.image(localfile)

    # if tryon2 and base_background_image and refs_background_image:
    #     localdir = f"users/{st.session_state.username}/images/{uuid.uuid4()}"
    #     os.makedirs(localdir, exist_ok=True)

    #     localfile = f"{localdir}/tryon.png"

    #     clothfile = f"{localdir}/cloth.png"
    #     clothmask = f"{localdir}/cloth_mask.png"
    #     personfile = f"{localdir}/person.png"
    #     personmask = f"{localdir}/person_mask.png"

    #     refs_background_image.save(clothfile)
    #     base_background_image.save(personfile)

    #     headers = {
    #         'Connection': 'keep-alive',
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    #     }
    #     data = {
    #         "person_image": personfile,
    #         "cloth_image": clothfile,
    #         "num_inference_steps": ddim_steps,
    #         "guidance_scale": scale,
    #         "seed": seed
    #     }
    #     files = {
    #         "person_image": ("person.png", open(personfile, 'rb')),
    #         "cloth_image": ("cloth.png", open(clothfile, 'rb')),
    #     }
    #     response = requests.post(url=f"http://{get_config('server_address')}/catvton/tryon_file", headers=headers, data=data, files=files)

    #     with open(localfile, "wb") as file:
    #         file.write(response.content)

    #     st.write("试穿效果图")
    #     st.image(localfile)
