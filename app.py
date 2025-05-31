import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка страницы
st.set_page_config(
    page_title="Прогноз отказов оборудования",
    page_icon="⚙️",
    layout="wide"
)

# Навигация
st.sidebar.title("⚙️ Прогноз отказов оборудования")
st.sidebar.write("---")

pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация проекта": presentation_page
}

selection = st.sidebar.radio("Навигация", list(pages.keys()))
st.sidebar.write("---")
st.sidebar.caption("Предиктивное обслуживание оборудования")

# Запуск выбранной страницы
pages[selection]()