import pandas as pd
import streamlit as st
from PIL import Image
from model import load_model_and_generate


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    #image = Image.open('data/titanic.jpg')
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Cringe Songs Generation",
        #page_icon=image,

    )
    st.write(
        """
        Generate your own song
        """
    )
    #st.image(image)


def write_generation_result(prediction_result):
    st.write("## Result")
    st.write(prediction_result)


def process_side_bar_inputs():
    st.sidebar.header('Input parameters')
    input_features = sidebar_input_features()
    try:
        generation_result = load_model_and_generate(input_features['model_type'], input_features['prompt'])
        write_generation_result(generation_result)
    except:
        write_generation_result("Exception. Another model")

def sidebar_input_features():
    model_type = st.sidebar.selectbox("Model", ("Self-made", "RuGPT finetuned"))
    prompt = st.text_input('Lyrics begin with:', 'Добрым словом')
    translateration = {
        "Self-made": "RMG_checkpoint.pkl",
        "RuGPT finetuned": "model_rugpt3large_gpt2_based.pkl",
    }
    data = {
        "model_type": translateration[model_type],
        "prompt": prompt,
    }
    return data


if __name__ == "__main__":
    process_main_page()
