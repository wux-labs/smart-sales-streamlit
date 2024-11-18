import streamlit as st
import streamlit_mermaid as stmd
import torch
from PIL import Image
import io
import os
import uuid
import json
import requests
import pandas as pd
import torch
import asyncio

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

from datetime import datetime
from database.database import engine
from sqlalchemy import text

from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_openai import OpenAI

from common.chat import get_default_client
from common.draw import save_draw_image
from common.product import select_product, display_products, product_vector_index, product_index_directory
from common.voice import text_to_voice, voice_to_text
# from common.talker import predict

from utils import init_page_header, init_session_state, get_avatar, get_config
from utils import global_system_prompt, main_container_height

from selenium import webdriver
from selenium.webdriver.common.by import By


title = "商品咨询"
icon = "🙋🏻"
init_page_header(title, icon)
init_session_state()

localdir = f"users/{st.session_state.username}/records"

conversation_system_prompt = """{global_system_prompt}

有一件{product_name}

衣服的亮点包括：{product_advantage}

衣服的其他详细信息包括：
{product_info}

请根据以上信息回答用户提出的问题，你需要精确获取到商品的亮点价值，激发用户的购买欲。

但请你铭记：禁止捏造数据！
"""

template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=f"{global_system_prompt}\n{template}", input_variables=["summaries", "question"]
)

tools = [{
            "type": "function",
            "function": {
                "name": "query_express",
                "description": "根据快递单号，查询快递信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "description": "快递单号",
                            "type": "string"
                        }
                    },
                    "required": [ "number" ]
                },
            }
        }, {
            "type": "function",
            "function": {
                "name": "query_weather",
                "description": "根据城市名称，查询城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "description": "城市名称",
                            "type": "string"
                        }
                    },
                    "required": [ "city" ]
                },
            }
        }, {
            "type": "function",
            "function": {
                "name": "draw_picture",
                "description": "根据用户的需求，生成一幅画",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "description": "用户的需求",
                            "type": "string"
                        }
                    },
                    "required": [ "prompt" ]
                },
            }
        }, {
            "type": "function",
            "function": {
                "name": "product_recommend",
                "description": "根据用户的需求，推荐合适的服装商品",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "description": "用户的需求",
                            "type": "string"
                        }
                    },
                    "required": [ "requirement" ]
                },
            }
        }, {
            "type": "function",
            "function": {
                "name": "flow_graph",
                "description": "根据用户的需求，提供做某件事情的流程",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "description": "用户的需求",
                            "type": "string"
                        }
                    },
                    "required": [ "requirement" ]
                },
            }
        }]


def query_express(number: str="YT8980437173209"):
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
    data={'number': number}
    response = requests.post(url=f'http://{get_config("server_address")}/tools/query_express', headers=headers, data=data)
    return response.json()


def query_weather(city):
    key=os.getenv("WEATHER_KEY", default="")

    location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
    city_code = None
    try:
        city_code_response = requests.get(
            location_query_url,
            params={'key': key, 'location': city}
        )
    except Exception as e:
        return str(e)
    if city_code_response.status_code != 200:
        return city_code_response.status_code, city_code_response.json()
    
    city_code_response = city_code_response.json()
    if len(city_code_response['location']) == 0:
        return '未查询到城市'
    city_code = city_code_response['location'][0]['id']
    weather_query_url = 'https://devapi.qweather.com/v7/weather/now'
    try:
        weather_response = requests.get(
            weather_query_url,
            params={'key': key, 'location': city_code}
        )
    except Exception as e:
        return str(e)

    now = weather_response.json()['now']
    data = [
        f'观测时间: {now["obsTime"]}',
        f'温度: {now["temp"]}°C',
        f'体感温度: {now["feelsLike"]}°C',
        f'天气: {now["text"]}',
        f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
        f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
        f'相对湿度: {now["humidity"]}',
        f'当前小时累计降水量: {now["precip"]} mm',
        f'大气压强: {now["pressure"]} 百帕',
        f'能见度: {now["vis"]} km',
    ]
    return "天气情况如下：\n  " + "\n  ".join(data)


def draw_picture(prompt):
    localfile = save_draw_image(user_input_text)
    return localfile


def product_recommend(requirement):
    vector_index = product_vector_index(product_index_directory)
    docs = vector_index.similarity_search(requirement, k=2)
    products = ",".join([str(doc.metadata.get("id")) for doc in docs])
    return products


def flow_graph(requirement):
    messages = [{
                    "role": "system",
                    "content": "你是一位优秀的服装商品智能销售专家，你熟悉电商平台的购物流程，并且你是精通Mermaid语法的专家。请根据用户的问题生成Mermaid格式的流程图，仅输出流程图，注意必须符合Mermaid的`graph LR`语法格式！别说废话。"
                }, {
                    "role": "user",
                    "content":requirement
                }]

    try:
        response = get_default_client().chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            stream=False,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.8
        ).choices[0].message.content

        return "graph" + response.split("graph")[1].replace("```", "")
    except:
        return ""

def load_product_documents(id):
    vector_index = product_vector_index(product_index_directory)
    # product_documents = vector_index.search("ALL","similarity", k=1, filter={"id":f"{id}"})
    product_documents = vector_index.search("ALL","similarity", k=1, filter=[{"match": {"_id":f"{id}"}}])
    return product_documents


def load_chain():
    chain = load_qa_with_sources_chain(
        llm=OpenAI(api_key="EMPTY", base_url=f'http://{get_config("server_address")}/v1/', model='glm-4-flash'),
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )
    return chain


def introduce_product(product_info):
    user_text = "你需要根据我给出的商品信息用500字文案详细描述一下这件服装，内容必须基于商品信息撰写，禁止捏造内容。文案中不要提及直播间，要说本店。文案中不要给出商品的任何链接，仅介绍商品信息。请仅输出文案内容，不要输出其它内容。你会和客户进行多轮会话，不要和客户说再见。"
    messages = [
        {"role": "system", "content": conversation_system_prompt.format(global_system_prompt=global_system_prompt, product_name=product_info.iloc[1], product_advantage=product_info.iloc[9], product_info=product_info.iloc[13])},
        {"role": "user", "content": user_text}
    ]
    answer = ""
    with main_container.chat_message("assistant", avatar=get_avatar("")):
        with st.spinner("处理中，请稍等..."):
            with st.empty():
                response = get_default_client().chat.completions.create(
                    model="glm-4-flash",
                    messages=messages,
                    stream=True,
                    max_tokens=2048,
                    temperature=0.8,
                    top_p=0.8,
                )
                if response:
                    for chunk in response:
                        if hasattr(chunk.choices[0].delta, "content"):
                            answer += chunk.choices[0].delta.content
                            st.markdown(answer)
                st.session_state["ask_product_history"].append({"role": "assistant", "content": answer, "audio": None, "video": None, "image": None, "product": None, "graph": None, "media": st.session_state["media_file"]})


def cache_ask_product(product_info, user_voice_file, user_input_text):
    user_input = ""
    localdir = f"users/{st.session_state.username}/records"
    with main_container.chat_message("user"):
        with st.spinner("处理中，请稍等..."):
            if user_voice_file is not None:
                st.audio(user_voice_file, format="wav")
                user_input = voice_to_text(localdir, filename)
                if st.session_state.config_assistant_display_text:
                    st.write(user_input)
            else:
                user_input = user_input_text
                st.write(user_input)

    try:
        with main_container.chat_message("assistant", avatar=get_avatar("")):
            with st.spinner("处理中，请稍等..."):
                audio_flag = False
                audio_path = None
                video_path = None
                image_path = None
                product_ids = None
                graph_content = None
                if "ask_product_media" not in st.session_state.keys():
                    stream=False if st.session_state.config_assistant_tools else True
                    messages = [
                                    {"role": "system", "content": conversation_system_prompt.format(global_system_prompt=global_system_prompt, product_name=product_info.iloc[1], product_advantage=product_info.iloc[9], product_info=product_info.iloc[11])},
                                    {"role": "system", "content": "不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息。"}
                                ]
                    for message in st.session_state["ask_product_history"][2:]:
                        if message["media"] == None:
                            messages.append({"role": message["role"], "content": message["content"] if message["content"] else ""})
                    messages.append({"role": "user", "content": user_input})

                    response = get_default_client().chat.completions.create(
                        model="glm-4-flash",
                        messages=messages,
                        stream=stream,
                        max_tokens=2048,
                        temperature=0.8,
                        top_p=0.8,
                        tools=tools if st.session_state.config_assistant_tools else None
                    )

                    empty_container = st.empty()

                    if stream:
                        answer = ""
                        with empty_container:
                            for chunk in response:
                                if hasattr(chunk.choices[0].delta, "content"):
                                    answer += chunk.choices[0].delta.content
                                    st.markdown(answer)
                            response = answer
                            audio_flag = True
                    elif response.choices[0].message.tool_calls is not None:
                        tool_call = response.choices[0].message.tool_calls[0]
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments

                        if function_name == "query_express":
                            response = query_express(**json.loads(function_args))
                        elif function_name == "query_weather":
                            response = query_weather(**json.loads(function_args))
                        elif function_name == "draw_picture":
                            image_path = draw_picture(**json.loads(function_args))
                            st.image(image_path)
                            response = None
                        elif function_name == "product_recommend":
                            product_ids = product_recommend(**json.loads(function_args))
                            display_products(product_ids)
                            response = None
                        elif function_name == "flow_graph":
                            graph_content = flow_graph(user_input)
                            stmd.st_mermaid(graph_content)
                            response = None
                        else:
                            response = response.choices[0].message.content
                    else:
                        response = response.choices[0].message.content

                    # else:
                    #     product_documents = load_product_documents(id)
                    #     chain = load_chain()
                    #     answer = chain(
                    #         {"input_documents": product_documents, "question": user_input}, return_only_outputs=True
                    #     )
                    #     response = answer["output_text"].split("SOURCES: ")[0]
                    #     audio_flag = True
                else:
                    headers = {
                        'Connection': 'keep-alive',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
                    }
                    files = {
                        "media": (media_file.name, open(st.session_state.ask_product_media, 'rb'))
                    }
                    data={
                        'media_type': "video" if st.session_state.ask_product_media.endswith(".mp4") else "image",
                        'user_input': user_input
                    }

                    response = requests.post(url=f'http://{get_config("server_address")}/vision_language/internvl/chat_with_media', headers=headers, data=data, files=files).json()
                    st.markdown(response)
                    audio_flag = True

                if audio_flag:
                    if st.session_state.config_talker_enable:
                        audio_path = text_to_voice(response)
                        st.audio(audio_path, format="wav")
                        if st.session_state.config_assistant_display_text:
                            st.write(response)
                        talker_audio_path = text_to_voice(response[:st.session_state.config_talker_limit])
                        headers = {
                            'Connection': 'keep-alive',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
                        }
                        data = {
                            'talker_name': st.session_state.config_talker_name
                        }
                        files = {
                            "audio": ("test.wav", open(talker_audio_path, 'rb'))
                        }
                        response = requests.post(url=f'http://{get_config("server_address")}/digital_human/{st.session_state.config_talker_model}/generate_video_file', headers=headers, data=data, files=files)

                        localdir = f"users/{st.session_state.username}/video"
                        os.makedirs(localdir, exist_ok=True)
                        video_path = f"{localdir}/{uuid.uuid4()}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(response.content)

                        talk_empty_container.video(video_path)
                        st.session_state["ask_product_talker"] = video_path
                    elif st.session_state.config_assistant_response_speech:
                        audio_path = text_to_voice(response)
                        st.audio(audio_path, format="wav")
                        if st.session_state.config_assistant_display_text:
                            st.write(response)
                elif response:
                        st.write(response)
                
                st.session_state["ask_product_history"].append({"role": "user", "content": user_input, "audio": user_voice_file, "video": None, "image": None, "product": None, "graph": None, "media": st.session_state["media_file"]})
                st.session_state["ask_product_history"].append({"role": "assistant", "content": response, "audio": audio_path, "video": video_path, "image": image_path, "product": product_ids, "graph": graph_content, "media": st.session_state["media_file"]})
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':

    id = None
    i = 0
    if "id" in st.query_params.keys():
        id = st.query_params["id"]
    elif "ask_product_id" in st.session_state.keys():
        id = st.session_state.ask_product_id
    if "media_file" not in st.session_state.keys():
        st.session_state["media_file"] = None
    
    if id is None:
        st.switch_page("pages/41🛍️商品管理.py")
    
    product_info = select_product(id).iloc[0]

    if "config_talker_enable" in st.session_state.keys() and st.session_state.config_talker_enable:
        cols = st.columns([0.55, 0.45])
        with cols[0]:
            main_container = st.container(height=main_container_height)
        with cols[1]:
            talk_container = st.container(height=main_container_height)
            talk_empty_container = talk_container.empty()
            if st.session_state["ask_product_talker"]:
                talk_empty_container.video(st.session_state["ask_product_talker"])
    else:
        main_container = st.container(height=main_container_height)

    with st.sidebar:
        tabs = st.tabs(["商品主图", "商品视频", "客服设置", "数字人"])
        with tabs[0]:
            if product_info.iloc[12]:
                st.image(product_info.iloc[12])
        with tabs[1]:
            if product_info.iloc[13]:
                st.video(product_info.iloc[13])
        with tabs[2]:
            cols = st.columns(2)
            with cols[0]:
                st.toggle("语音回复", key="config_assistant_response_speech")
                st.toggle("智能工具", key="config_assistant_tools")
            with cols[1]:
                st.toggle("显示文字", key="config_assistant_display_text")
            media_file = st.file_uploader(label="视觉物料", key="config_media_file", type=["jpg", "png", "mp4"])
            if media_file:
                if "ask_product_media" not in st.session_state.keys():
                    localdir = f"users/{st.session_state.username}/upload/{datetime.now().strftime('%Y-%m-%d')}/{uuid.uuid4()}"
                    os.makedirs(localdir, exist_ok=True)

                    with open(f"{localdir}/{media_file.name}", "wb") as f:
                        f.write(media_file.getvalue())

                    st.session_state["ask_product_media"] = f"{localdir}/{media_file.name}"
                    st.session_state["media_file"] = media_file.name
                    
                    if st.session_state.ask_product_media.endswith(".mp4"):
                        st.session_state["ask_product_history"].append({"role": "user", "content": None, "audio": None, "video": st.session_state.ask_product_media, "image": None, "product": None, "graph": None, "media": st.session_state["media_file"]})
                    else:
                        st.session_state["ask_product_history"].append({"role": "user", "content": None, "audio": None, "video": None, "image": st.session_state.ask_product_media, "product": None, "graph": None, "media": st.session_state["media_file"]})
            elif "ask_product_media" in st.session_state.keys():
                del st.session_state["ask_product_media"]
        with tabs[3]:
            st.toggle("启用数字人", key="config_talker_enable")
            limit = st.slider("", key="config_talker_limit", min_value=1, max_value=100, value=15, step=1)
            cols = st.columns(2)
            with cols[0]:
                model = st.selectbox("Model", key="config_talker_model", options=["hallo", "v-express"])
            if model == "hallo":
                with cols[1]:
                    talker_name = st.selectbox("People", key="config_talker_name", options=range(1,7))
                st.image(f"statics/digital_human/hallo/{talker_name}.jpg")
            if model == "v-express":
                with cols[1]:
                    talker_name = st.selectbox("People", key="config_talker_name", options=["kara", "tys", "yomir"])
                st.image(f"statics/digital_human/v_express/{talker_name}/ref.jpg")
        cols = st.columns(5)
        with cols[2]:
            audio_bytes = audio_recorder(text="", pause_threshold=2.5, icon_size='2x', sample_rate=16000)

    if "ask_product_history" not in st.session_state.keys():
        st.session_state["ask_product_history"] = [{
                                                        "role": "system",
                                                        "content": conversation_system_prompt.format(global_system_prompt=global_system_prompt, product_name=product_info.iloc[1], product_advantage=product_info.iloc[9], product_info=product_info.iloc[11]),
                                                        "audio": None,
                                                        "video": None,
                                                        "image": None,
                                                        "product": None,
                                                        "graph": None,
                                                        "media": st.session_state["media_file"]
                                                    }]

    if "ask_product_talker" not in st.session_state.keys():
        st.session_state["ask_product_talker"] = None

    for message in st.session_state["ask_product_history"]:
        content = message['content']
        audio = message["audio"]
        video = message["video"]
        image = message["image"]
        product = message["product"]
        graph = message["graph"]
        if message['role'] == 'user':
            with main_container.chat_message("user"):
                if audio:
                    st.audio(audio, format="audio/mp3")
                    if st.session_state.config_assistant_display_text:
                        st.write(content)
                elif image:
                    st.image(image, width=512)
                elif video:
                    cols = st.columns([0.3, 0.7])
                    with cols[0]:
                        st.video(video)
                elif content:
                    st.write(content)
        elif message['role'] == 'assistant':
            with main_container.chat_message("assistant", avatar=get_avatar("")):
                if audio:
                    st.audio(audio, format="audio/mp3")
                    if st.session_state.config_assistant_display_text:
                        st.write(content)
                elif image:
                    st.image(image, width=512)
                elif video:
                    cols = st.columns([0.5, 0.5])
                    with cols[0]:
                        st.video(video)
                elif product:
                    i = i + 1
                    display_products(product, nowtime=i)
                elif graph:
                    stmd.st_mermaid(graph)
                elif content:
                    st.write(content)

    if len(st.session_state["ask_product_history"]) == 1:
        introduce_product(product_info)

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        cache_ask_product(product_info, user_voice_file=None, user_input_text=user_input_text)
    elif audio_bytes:
        os.makedirs(localdir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.wav"
        filepath = f"{localdir}/{filename}"

        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        audio_segment.export(filepath, format='wav')
        cache_ask_product(product_info, user_voice_file=filepath, user_input_text=None)
