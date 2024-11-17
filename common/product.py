import streamlit as st
import pandas as pd
import os
import subprocess
import json
import uuid
import requests
from datetime import datetime

from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore


from database.database import engine
from sqlalchemy import text
from datetime import datetime

from utils import get_config, image_to_base64, update_aigc_perm_freq


product_index_directory = "products/product_index"


def image_chat_answer(localdir, filename):
    headers = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
    data={
        'product_name': st.session_state.input_product_name,
        'product_tags': st.session_state.input_product_tags,
        'product_gender': st.session_state.input_product_gender,
        'product_season': st.session_state.input_product_season,
        'product_price': st.session_state.input_product_price,
        'product_style': st.session_state.input_product_style,
        'product_material': st.session_state.input_product_material,
        'product_advantage': st.session_state.input_product_advantage,
        'product_description': st.session_state.input_product_description
    }
    files = {
        "image": (filename, open(f"{localdir}/{filename}", 'rb'))
    }
    response = requests.post(url=f"http://{get_config('server_address')}/vision_language/qwen2vl/marketing_documents", headers=headers, data=data, files=files)

    return response.json()


@st.cache_resource
def load_huggingface_embedding():
    embedding = HuggingFaceEmbeddings(model_name="models/GanymedeNil/text2vec-large-chinese")
    return embedding


def product_vector_index(persist_directory="products/product_index"):
    embedding = load_huggingface_embedding()
    elastic_vector_index = ElasticsearchStore(es_url=f"http://{get_config('elasticsearch_server')}", index_name="product_index", embedding=embedding) 
    return elastic_vector_index


def select_product(ids):
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            select id as 商品编号, name as 商品名称, title as 商品标题, tags as 商品标签,
                   gender as 商品类型, season as 适合季节, price as 商品价格,
                   style as 设计风格, material as 服装材质, advantage as 服装亮点,
                   marketing as 种草文案, description as 商品描述,
                   image as 商品主图, video as 商品视频
              from ai_labs_product_info
             where id in ({ids})
            """, conn)
        return df


def save_product_ratings(product_id, rating, comment):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_product_ratings(user_id, product_id, rating, comment, date_time)
            values(:user_id, :product_id, :rating, :comment, :date_time)
            ''')
            conn.execute(sql, [{
                'user_id': st.session_state.userid,
                'product_id': product_id,
                'rating': rating,
                'comment': comment,
                'date_time': date_time
            }])
            conn.commit()

            st.toast("评价成功！", icon="✌️")

    except Exception as e:
        st.exception(e)


def display_products(ids, nowtime=uuid.uuid4()):
    products = select_product(ids)
    products["商品主图"]=products["商品主图"].apply(lambda x: "data:image/png;base64," + image_to_base64(x))

    for _, row in products.iterrows():
        with st.expander(row.iloc[2]):
            cols = st.columns(3)
            with cols[0]:
                tabs = st.tabs(["商品主图", "商品视频"])
                with tabs[0]:
                    if row.iloc[12]:
                        st.image(row.iloc[12])
                with tabs[1]:
                    if row.iloc[10]:
                        st.video(row.iloc[13])
            with cols[1]:
                st.write("商品名称：" + row.iloc[1])
                st.write("商品标签：" + row.iloc[3])
                st.write("商品类型：" + row.iloc[4])
                st.write("适合季节：" + row.iloc[5])
                st.write("商品价格：" + str(row.iloc[6]))
                st.write("设计风格：" + row.iloc[7])
                st.write("服装材质：" + row.iloc[8])
                st.write("服装亮点：" + row.iloc[9])
            with cols[2]:
                st.markdown(row.iloc[11])
            st.markdown("-------")
            st.markdown(str(row.iloc[10]))
            st.markdown("-------")
            cols = st.columns([0.2,0.2,0.2,0.2,0.2])
            with cols[0]:
                if st.button("👚在线试穿",  key=f"try_on_{row.iloc[0]}_{nowtime}"):
                    st.session_state.tryon_id = row.iloc[0]
                    st.switch_page("pages/61👚在线试穿.py")
            with cols[1]:
                if st.button("🙋🏻商品咨询", key=f"ask_product_{row.iloc[0]}_{nowtime}"):
                    st.session_state.ask_product_id = row.iloc[0]
                    st.switch_page("pages/42🙋🏻商品咨询.py")
            with cols[2]:
                if st.button("💰立即购买", key=f"buy_product_{row.iloc[0]}_{nowtime}"):
                    update_aigc_perm_freq(int(row.iloc[6] * 10))
                    st.toast("购买成功，感谢您的支持！", icon="💰")
            with cols[3]:
                if st.button("🛒加购物车", key=f"favorite_product_{row.iloc[0]}_{nowtime}"):
                    st.toast("加购成功，感谢您的支持！", icon="🛒")
            with cols[4]:
                with st.popover("📝我要评价"):
                    with st.form(f"comment_product_form_{row.iloc[0]}_{nowtime}"):
                        rating = st.number_input("评分", key=f"product_rating_{row.iloc[0]}_{nowtime}", min_value=1, max_value=5)
                        comment = st.text_area(label="评论", key=f"product_comment_{row.iloc[0]}_{nowtime}")
                        if st.form_submit_button("提交", type="primary", use_container_width=True):
                            save_product_ratings(row.iloc[0], rating, comment)
