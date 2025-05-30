import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Навигация
pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти", list(pages.keys()))
page = pages[selection]
page()