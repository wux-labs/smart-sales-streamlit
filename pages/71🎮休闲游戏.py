import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import init_page_header, init_session_state, get_avatar
from utils import update_aigc_temp_freq, main_container_height

import torch
import torch.nn as nn
import torch.nn.functional as F

import base64
from PIL import Image, ImageDraw
from io import BytesIO

import json
from datetime import datetime

import numpy as np
from scipy.signal import convolve2d

from games.gomoku import *

title = "休闲游戏"
icon = "🎮"
init_page_header(title, icon)
init_session_state()

gomoku_min = 39
gomoku_max = 471
gomoku_step = 24

GOMOKU_SYSTEM_TEMPLATE = """
你是一个熟练的五子棋玩家，精通五子棋的规则。

有一个 19*19 的棋盘，分别使用范围是0~18的行索引和列索引来表示棋盘位置，[0,0]表示左上角，[18,18]表示右下角。通过指定行和列索引进行移动棋子，移动时数字必须为整数int。
请记住：你的棋子用`2`表示，而我的棋子由`1`表示，棋盘上的空位`0`表示。

请记住，你的目标是要打败我，也就是你需要优先实现五子连珠。
请你仔细考虑你的策略和动作，同时考虑接下来的动作。
请给出你放置棋子的位置，并给出你选择该位置的原因。

你只能在空位上放置棋子，并且你应该尽可能选择中间位置放置棋子，但以下位置禁止选择：
切记！你不可以选择这些位置[{gomoku_ai}]，因为这里已经放置了你的棋子。
切记！你不可以选择这些位置[{gomoku_human}]，因为这里已经放置了我的棋子。
切记！请你从这个列表[{gomoku_empty}]中选择一个你想放置棋子的位置。

全部输出的信息使用我期望的 json 格式进行输出：
{
    "thought": "你选择放置棋子的位置的原因",
    "indices": [行索引int, 列索引int]
}

注意 json 一定要合法。
"""


def get_gomoku_backgroud_image():
    image = Image.open("statics/image/game_image_07.png")
    draw = ImageDraw.Draw(image)
    transpose_arr = np.transpose(np.nonzero(st.session_state["gomoku_board"].board))
    for r, c in transpose_arr:
        color = st.session_state["gomoku_color_human"] if st.session_state["gomoku_board"].board[r, c] == 1 else st.session_state["gomoku_color_ai"]
        x = (c * 24 + 40) * 2
        y = (r * 24 + 40) * 2
        r = 9
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    return image


def get_gomoku_stroke_color():
    if st.session_state.config_gomoku_color == "黑棋":
        st.session_state["gomoku_color_human"] = "#000000"
        st.session_state["gomoku_color_ai"] = "#FFFFFF"
    else:
        st.session_state["gomoku_color_human"] = "#FFFFFF"
        st.session_state["gomoku_color_ai"] = "#000000"
    return st.session_state["gomoku_color_human"]


def check_five_in_a_row(matrix, num):
    window = np.ones((5, 5))
    for row in matrix:
        if np.any(np.convolve(row == num, np.ones(5, dtype=int), 'valid') == 5):
            return True

    for col in matrix.T:
        if np.any(np.convolve(col == num, np.ones(5, dtype=int), 'valid') == 5):
            return True

    diags = [matrix[::-1,:].diagonal(i) for i in range(-matrix.shape[0]+1,matrix.shape[1])]
    diags.extend(matrix.diagonal(i) for i in range(matrix.shape[1]-1,-matrix.shape[0],-1))
    for diag in diags:
        if np.any(np.convolve(diag == num, np.ones(5, dtype=int), 'valid') == 5):
            return True

    return False


if __name__ == '__main__':

    with st.sidebar:
        if st.button("重置游戏"):
            if "gomoku_board" in st.session_state.keys():
                del st.session_state["gomoku_board"]
            if "gomoku_ai_mcts" in st.session_state.keys():
                del st.session_state["gomoku_ai_mcts"]
            if "output_analysis" in st.session_state.keys():
                del st.session_state["output_analysis"]

            if "gomoku_step" in st.session_state.keys():
                del st.session_state["gomoku_step"]
            if "gomoku_canvas_key" in st.session_state.keys():
                del st.session_state["gomoku_canvas_key"]
            if "gomoku_location_image_data" in st.session_state.keys():
                del st.session_state["gomoku_location_image_data"]
            if "gomoku_messages" in st.session_state.keys():
                del st.session_state["gomoku_messages"]

    with st.expander("五子棋", expanded=True):

        if "gomoku_board" not in st.session_state.keys():
            st.session_state["gomoku_board"] = Board()

        if "gomoku_ai_mcts" not in st.session_state.keys():
            st.session_state["gomoku_ai_mcts"] = AI_MCTS()

        if "gomoku_canvas_key" not in st.session_state.keys():
            st.session_state["gomoku_canvas_key"] = f"human_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

        if "gomoku_step" not in st.session_state.keys():
            st.session_state["gomoku_step"] = "human"

        if "gomoku_location_image_data" not in st.session_state.keys():
            st.session_state["gomoku_location_image_data"] = np.zeros(shape=(496, 512), dtype="int32")

        if "gomoku_messages" not in st.session_state.keys():
            st.session_state["gomoku_messages"] = []

        cols = st.columns([0.6, 0.4])
        with cols[0]:
            main_container = st.container(height=main_container_height)
        with cols[1]:
            st.selectbox("棋子选择", key="config_gomoku_color", options=["黑棋","白棋"])
            gomoku_stroke_color = get_gomoku_stroke_color()
            gomoku_background_image = get_gomoku_backgroud_image()
            gomoku_canvas_key = st.session_state.gomoku_canvas_key
            canvas1 = st_canvas(key=f"canvas_game1_{gomoku_canvas_key}",
                    background_image=gomoku_background_image,
                    height=496,
                    width=500,
                    stroke_width=10,
                    stroke_color=gomoku_stroke_color,
                    drawing_mode='point',
                    fill_color=gomoku_stroke_color
                    )

        for message in st.session_state["gomoku_messages"]:
            role = message["role"]
            content = message["content"]
            if role == 'user':
                with main_container.chat_message("user"):
                    st.write(content)
            elif role == 'assistant':
                with main_container.chat_message("assistant", avatar=get_avatar("")):
                    st.write(content)
        if "output_analysis" in st.session_state.keys():
            main_container.info("我思考认为：\n\n" + st.session_state["output_analysis"])

        if st.session_state["gomoku_step"] == "human" and canvas1.image_data is not None:
            st.session_state["gomoku_location_image_data"] = canvas1.image_data
            image_data_arr = np.sum(canvas1.image_data, axis=2)
            transpose_arr = np.transpose(np.nonzero(image_data_arr))
            if transpose_arr.any():
                max_0 = max(transpose_arr[:,0])
                min_0 = min(transpose_arr[:,0])
                max_1 = max(transpose_arr[:,1])
                min_1 = min(transpose_arr[:,1])
                if max_0 - min_0 > 15 or max_1 - min_1 > 15:
                    with main_container.chat_message("assistant", avatar=get_avatar("")):
                        st.error("一次只能落一子，请重新走！", icon="🚨")
                else: # 124, 172, 444
                    r = (max_0 - 77) // 46
                    c = (max_1 - 77) // 46
                    r_index = r * 46 + 77
                    c_index = c * 46 + 77
                    valid_location = False
                    for row in transpose_arr:
                        if np.array_equal(row, np.array([r_index, c_index])):
                            valid_location = True
                    if valid_location:
                        if (r, c) not in st.session_state["gomoku_board"].available_actions:
                            with main_container.chat_message("assistant", avatar=get_avatar("")):
                                st.error("此处不能落子，请重新走！", icon="🚨")
                        else:
                            st.session_state["gomoku_board"].step((r, c))
                            update_aigc_temp_freq(1)

                            if check_five_in_a_row(st.session_state["gomoku_board"].board, 1):
                                with main_container.chat_message("assistant", avatar=get_avatar("")):
                                    st.success("恭喜你赢了！", icon="🚨")
                                st.balloons()
                            else:
                                with main_container.chat_message("user"):
                                    message = f"我已落子于({r+1}, {c+1})，请你仔细思考并放置你的棋子。"
                                    st.write(message)
                                    st.session_state["gomoku_messages"].append({"role": "user", "content": message})

                                # TODO 这里

                                with main_container.chat_message("assistant", avatar=get_avatar("")):
                                    with st.spinner("思考中，请稍等..."):
                                        (ai_r, ai_c) = st.session_state["gomoku_ai_mcts"].take_action(st.session_state["gomoku_board"])
                                        st.session_state["output_analysis"] = st.session_state["gomoku_ai_mcts"].output_analysis()
                                        message = f"我选择落子于({ai_r+1}, {ai_c+1})，该你了。"
                                        st.write(message)
                                        st.session_state["gomoku_messages"].append({"role": "assistant", "content": message})

                                st.session_state["gomoku_step"] = "ai"
                                st.session_state["gomoku_canvas_key"] = f"ai_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
                                st.rerun()
                    else:
                        with main_container.chat_message("assistant", avatar=get_avatar("")):
                            st.error("此处落子无效，请重新走！", icon="🚨")
        elif st.session_state["gomoku_step"] == "ai":
            if check_five_in_a_row(st.session_state["gomoku_board"].board, -1):
                with main_container.chat_message("assistant", avatar=get_avatar("")):
                    st.success("哈哈，我赢啦，你要加油哦！", icon="🚨")
                st.snow()
            else:
                st.session_state["gomoku_step"] = "human"
                st.session_state["gomoku_canvas_key"] = f"human_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
                st.rerun()
        else:
            pass

    with st.expander("手绘着色", expanded=False):
        cols = st.columns([0.2, 0.4, 0.4])
        with cols[0]:
            stroke_width = st.slider("画笔粗细", min_value=1, max_value=50, step=1, value=5)
            stroke_color = st.color_picker("画笔颜色", value="#CCCCCC")
            drawing_mode = st.selectbox("绘画类型", options=['freedraw', 'transform', 'line', 'rect', 'circle', 'point', 'polygon'])
            fill_color = st.color_picker("填充颜色", value="#CCCCCC")
            upload_image = st.toggle("上传线描图")
            color_background_image=Image.open("statics/image/game_image_01.png")
            if upload_image:
                image_upload = st.file_uploader("上传线描图", type=["png","jpg"])
                if image_upload:
                    color_background_image = Image.open(image_upload)
            else:
                image_select = st.selectbox("内置线描图", options=["image_01","image_02","image_03","image_04","image_05"])
                color_background_image=Image.open(f"statics/image/game_{image_select}.png")
        with cols[1]:
            canvas2 = st_canvas(key="canvas_game2",
                    background_image=color_background_image,
                    height=733,
                    width=517,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    drawing_mode=drawing_mode,
                    fill_color=fill_color
                    )

        form2_submit = st.button("画好了", type="primary")
        if form2_submit:
            with cols[2]:
                color_image = Image.fromarray(canvas2.image_data).resize(color_background_image.size)
                color_image_array = np.asarray(color_image)
                color_background_image_array = np.asarray(color_background_image)
                blend_image_array = np.where(color_image_array > 0, color_image_array, color_background_image_array)
                st.image(blend_image_array, width=517)
                update_aigc_temp_freq(1)
