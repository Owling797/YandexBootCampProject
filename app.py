import pandas as pd
import streamlit as st
from PIL import Image
from model import load_model_and_generate
import requests


FASTAPI_URL = 'http://localhost:8000'


headers = {"Authorization": "Bearer *****"}

def process_main_page():
    show_main_page()
    #process_side_bar_inputs()


def show_main_page():
    #image = Image.open('data/titanic.jpg')
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Cringe Songs Generation",
        #page_icon=image,

    )
    st.write("# Generate your own song")
    #st.image(image)
    st.header('Input parameters')
    model_name = st.selectbox("Model", ("Self-made", "RuGPT finetuned"))
    prompt = st.text_input('Lyrics begin with:', 'Добрым словом')
    generate_button = st.button("Generate")
    if generate_button:
        input_features = {
            "model_name": model_name,
            "input_text": prompt,
        }
        generate_text(input_features)

def generate_text(input_features):
    url = f"{FASTAPI_URL}/generate_text/"
    response = requests.get(url, params=input_features)
    if response.status_code == 200:
        generation_result = response.json()['generated_text']
        write_generation_result(generation_result)
    else:
        st.error("Error 😕")
        
def write_generation_result(prediction_result):
    st.write("## Result")
    st.write(prediction_result)

if __name__ == "__main__":
    process_main_page()
