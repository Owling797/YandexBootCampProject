import pandas as pd
import streamlit as st
import  streamlit_toggle as tog
from PIL import Image
import requests
from streamlit_extras.stylable_container import stylable_container


FASTAPI_URL = 'http://localhost:8000'


headers = {"Authorization": "Bearer *****"}
#st.set_theme('dark')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def process_main_page():
    #print(help(tog.st_toggle_switch), dir(tog.st_toggle_switch))
    show_main_page()


def show_main_page():
    #image = Image.open('data/Bold Beats.gif')
    #video_file = open('data/video.mp4', 'rb')
    #video_bytes = video_file.read()
    st.set_page_config(
            layout="wide",
            initial_sidebar_state="auto",
            page_title="Cringe Songs Generation",
            #page_icon=image,

        )
    
    local_css("style.css")
    col1, col2, col3 = st.columns(3)
    # #0E1117 Background color #273346
    # #262730 Secondary background color #B9F1C0
    # 7792E3
    #[theme]
    #base="dark"
    #primaryColor="#4bffbd"
        
    with col2:
        with st.container(height=600, border=None):

            st.header("Lyrics & Song generator")
            #st.video(video_bytes)
            #st.image(image, use_column_width=True, clear_cache=True)
            #st.header('Input parameters')
            st.write("##### Model:")
            model_name = st.radio("Model", ("Self-made (no prompt)", "RuGPT finetuned"), label_visibility='collapsed')
            with st.container(border=True):
                chastushka_audio_on = st.checkbox('Davay nashu (folklore)')
                voice_on = st.checkbox('Pronounce')
                audio_on = st.checkbox('Generate song!')
            
            if audio_on:
                genre = st.selectbox('Select song genre', ('Rock', 'Pop', 'Folk'))

            if model_name == "RuGPT finetuned":
                st.write("\n##### Lyrics begin with:")
                with st.container(border=True):
                    prompt = st.text_input('Lyrics begin with:', 'Добрым словом', label_visibility='collapsed')
            else:
                prompt = ""
                
                
            
            generate_button = st.button("Generate")
            translateration = {
                "Self-made": "RMG_checkpoint.pkl",
                "RuGPT finetuned": "model_rugpt3large_gpt2_based.pkl",
            }
            if generate_button:
                input_features = {
                    "model_name": translateration[model_name],
                    "input_text": prompt,
                }
                generation_result = generate_text(input_features)
                if generation_result is not None:
                    write_generation_result(generation_result)
                    if voice_on:
                        get_tts_result_audio(generation_result)
                    if chastushka_audio_on:
                        get_chastushka_audio(generation_result)
                    if audio_on:
                        generate_suno_audio(generation_result, genre)

def generate_text(input_features):
    url = f"{FASTAPI_URL}/generate_text/"
    response = requests.get(url, params=input_features)
    if response.status_code == 200:
        generation_result = response.json()['generated_text']
        return generation_result
        #write_generation_result(generation_result)
    else:
        st.error("Error 😕")
        return None
        
def write_generation_result(prediction_result):
    st.write("\n#### Result")
    with st.container(height=200, border=True):
        st.write(prediction_result)
        
def get_tts_result_audio(text):
    pass

def get_chastushka_audio(text):
    pass

def generate_suno_audio(text, genre):
    pass

if __name__ == "__main__":
    process_main_page()
    
