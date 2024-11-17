import streamlit as st
from utils import init_page_header, init_session_state

title = "系统介绍"
icon = "📖"
init_page_header(title, icon)
init_session_state()


if __name__ == '__main__':
    cols = st.columns(2)
    with cols[0]:
        st.image("statics/doc/doc_image_01.png")
    with cols[1]:
        st.image("statics/doc/doc_image_02.png")
    tabs = st.tabs(["获客","活客","留客"])
    with tabs[0]:
        st.image("statics/doc/doc_image_03.png")
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/doc/image_01.png")
        with cols[1]:
            st.image("statics/doc/image_02.png")
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/doc/image_03.png")
        with cols[1]:
            st.image("statics/doc/image_04.png")
    with tabs[1]:
        st.image("statics/doc/doc_image_04.png")
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/doc/image_05.png")
        with cols[1]:
            st.image("statics/doc/image_06.png")
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/doc/image_07.png")
        with cols[1]:
            st.image("statics/doc/image_08.png")
    with tabs[2]:
        st.image("statics/doc/doc_image_05.png")
        st.image("statics/doc/doc_image_06.png")
