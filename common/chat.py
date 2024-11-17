import streamlit as st
import openai
import os

import torch
from transformers import AutoTokenizer

from utils import get_config, internlm_online_models, zhipuai_online_models, local_models, global_system_prompt

from openai import OpenAI

deepinfra_models = ["meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf",
                    "codellama/CodeLlama-34b-Instruct-hf", "jondurbin/airoboros-l2-70b-gpt4-1.4.1",
                    "mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

deepinfra_client = OpenAI(api_key="", base_url=os.getenv("DEEPINFRA_OPENAI_BASE", ""))

def init_chat_config_form():

    model = st.selectbox("Model", key="config_chat_model", options=local_models)
    max_tokens = st.number_input("Max Tokens", key="config_chat_max_tokens", min_value=512, max_value=4096,
                                    step=1, value=2048,
                                    help="The maximum number of tokens to generate in the chat completion.The total length of input tokens and generated tokens is limited by the model's context length.")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", key="config_chat_temperature", min_value=0.1, max_value=2.0,
                                value=1.0, step=0.1,
                                help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
        presence_penalty = st.slider("Presence Penalty", key="config_chat_presence_penalty", min_value=-2.0,
                                        max_value=2.0, value=0.0, step=0.1,
                                        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    with col2:
        top_p = st.slider("Top P", key="config_chat_top_p", min_value=1.0, max_value=5.0, value=1.0, step=1.0,
                            help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.")
        frequency_penalty = st.slider("Frequency Penalty", key="config_chat_frequency_penalty", min_value=-2.0,
                                        max_value=2.0, value=0.0, step=0.1,
                                        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    system_prompt = st.text_area("System Prompt", key="config_chat_system_prompt", value=global_system_prompt)


def get_default_client():
    return OpenAI(api_key="EMPTY", base_url=f'http://{get_config("server_address")}/v1/')

def get_chat_client():
    if st.session_state.config_chat_model in internlm_online_models:
        return OpenAI(api_key=os.getenv("INTERNLM_TOKEN", default=""), base_url=f'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/')
    elif st.session_state.config_chat_model in zhipuai_online_models:
        return OpenAI(api_key=os.getenv("CHATGLM_TOKEN", default=""), base_url=f'https://open.bigmodel.cn/api/paas/v4/')
    else:
        return OpenAI(api_key="EMPTY", base_url=f'http://{get_config("server_address")}/{st.session_state.config_chat_model}/v1/')
