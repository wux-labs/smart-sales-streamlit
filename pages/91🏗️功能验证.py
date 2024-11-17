import streamlit as st
import torch
import gc
import os, subprocess

from utils import init_page_header, init_session_state
from PIL import Image


title = "功能验证"
icon = "🏗️"
init_page_header(title, icon)
init_session_state()


def clear_streamlit_cache(keeps):
    all_caches = ["chat_tokenizer", "chat_model", 
                  "stable_diffusion_model",
                  "xcomposer2_vl_tokenizer", "xcomposer2_vl_model",
                  "whisper_model_base", "whisper_model_small", "whisper_model_medium", "whisper_model_large",
                  "ask_product_history", "ask_product_llm",
                  "sales_agent_model"
                  ]

    for cache in all_caches:
        if cache not in keeps and cache in st.session_state.keys():
            del st.session_state[cache]

    torch.cuda.empty_cache()

if st.button("清理缓存"):
    clear_streamlit_cache([""])
    torch.cuda.empty_cache()

cmd_text = st.chat_input("您的输入...")

if cmd_text:
    st.info(subprocess.getoutput(cmd_text).replace("\n", "  \n"))
