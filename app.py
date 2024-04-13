import time
import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import json
import base64


FASTAPI_URL = 'https://5a1e-2a0d-5600-1b-4000-29b1-f3ac-b68e-bd02.ngrok-free.app' # 'http://localhost:8000'
INPUT_TEXT_RES=""

headers = {"Authorization": "Bearer *****"}
#st.set_theme('dark')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def process_main_page():
    #print(help(tog.st_toggle_switch), dir(tog.st_toggle_switch))
    show_main_page()


def show_main_page():
    global INPUT_TEXT_RES
    print(INPUT_TEXT_RES)
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
            #with st.form(key='my_form', border=False):

                st.header("Lyrics & Song generator")
                #st.video(video_bytes)
                #st.image(image, use_column_width=True, clear_cache=True)
                #st.header('Input parameters')
                st.write("##### Model:")
                model_name = st.radio("Model", ("Self-made (no prompt)", "RuGPT finetuned"), label_visibility='collapsed')
                with st.container(border=True):
                    generate_text_pref = st.checkbox('Get Lyrics', value=True)
                    chastushka_audio_on = st.checkbox('Davay nashu (folklore)')
                    voice_on = st.checkbox('Pronounce')
                    audio_on = st.checkbox('Generate song!')
                
                if audio_on:
                    options = ['','Appalachian',
                                'Bluegrass',
                                'Country',
                                'Folk',
                                'Freak Folk',
                                'Western',
                                'Afro-Cuban',
                                'Dance Pop',
                                'Disco',
                                'Dubstep',
                                'Disco Funk',
                                'EDM',
                                'Electro',
                                'High-NRG',
                                'House',
                                'Trance',
                                'Ambient',
                                'Downtempo',
                                'Synthwave',
                                'Trap',
                                'Ambient',
                                'Cyberpunk',
                                'Drum n bass',
                                'Dubstep',
                                'Electronic',
                                'Hypnogogical',
                                'IDM',
                                'Phonk',
                                'Synthpop',
                                'Techno',
                                'Trap',
                                'Jazz/Soul',
                                'Bebop',
                                'Gospel',
                                'Electro',
                                'Frutiger Aero',
                                'Jazz',
                                'Latin Jazz',
                                'RnB',
                                'Soul',
                                'Latin',
                                'Bossa Nova',
                                'Latin Jazz',
                                'Mambo',
                                'Salsa',
                                'Tango',
                                'Reggae',
                                'Afrobeat',
                                'Dancehall',
                                'Dub',
                                'Reggae',
                                'Reggaeton',
                                'Metal',
                                'Black Metal',
                                'Deathcore',
                                'Death Metal',
                                'Heavy Metal',
                                'Heavy Metal Trap',
                                'Metalcore',
                                'Nu Metal',
                                'Power Metal',
                                'Pop',
                                'Dance Pop',
                                'Pop Rock',
                                'Kpop',
                                'Jpop',
                                'RnB',
                                'Synthpop',
                                'Rock',
                                'Classic Rock',
                                'Blues Rock',
                                'Emo',
                                'Glam Rock',
                                'Hardcore Punk',
                                'Indie',
                                'Industrial Rock',
                                'Punk',
                                'Rock',
                                'Skate Rock',
                                'Skatecore',
                                'Suomipop',
                                'Urban',
                                'Funk',
                                'Electro',
                                'HipHop',
                                'RnB',
                                'Phonk',
                                'Rap',
                                'Trap']

                    # Добавляем возможность ввода значения, которого нет в списке
                    st.write("Выберите или введите значение")
                    genre_input = st.text_input('Genre', '', label_visibility='collapsed', placeholder='type some genre')
                    genre = st.selectbox("Выберите или введите значение", options, label_visibility='collapsed')

                if model_name == "RuGPT finetuned":
                    st.write("\n##### Lyrics begin with:")
                    with st.container(border=True):
                        prompt = st.text_input('Lyrics begin with:', 'Добрым словом', label_visibility='collapsed')
                else:
                    prompt = ""
                    
                    
                
                generate_button = st.button("Generate")
                #generate_button = st.form_submit_button(label='Generate', use_container_width=True)
                translateration = {
                    "Self-made (no prompt)": "RMG_alpha_finetuned.pkl",
                    "RuGPT finetuned": "model_rugpt3large_gpt2_based.pkl",
                }
                if generate_button:
                    input_features = {
                        "model_name": translateration[model_name],
                        "input_text": prompt,
                    }
                    # 
                    if generate_text_pref:
                        print("text gen")
                        generation_result = generate_text(input_features)
                        INPUT_TEXT_RES = generation_result
                        if generation_result is not None:
                            write_generation_result(generation_result)
                            if voice_on:
                                get_tts_result_audio(generation_result)
                            if chastushka_audio_on:
                                get_chastushka_audio(generation_result)
                            if audio_on:
                                genre = genre if genre_input == "" else genre_input
                                generate_suno_audio(generation_result, genre)
                    else:
                        print("music gen")
                        if voice_on:
                            get_tts_result_audio(prompt)
                        if chastushka_audio_on:
                            get_chastushka_audio(prompt)
                        if audio_on:
                            genre = genre if genre_input == "" else genre_input
                            generate_suno_audio(prompt, genre)
                elif INPUT_TEXT_RES != "":
                    write_generation_result(INPUT_TEXT_RES)

@st.cache_data
def generate_text(input_features):
    #url = "https://ed45-178-154-246-234.ngrok-free.app/RMG_prediction"
    #response = requests.post(url)
    #generation_result = response.text
    #write_generation_result(generation_result)
        
    url = f"{FASTAPI_URL}/generate_text/"
    try:
        response = requests.get(url, params=input_features)
        #time.sleep(60*10)
        #if response.status_code != 200:
        #    time.sleep(60*3)
        #response.raise_for_status()  # Проверка наличия ошибок при выполнении запроса
        # Продолжайте выполнение кода здесь, используя response
        #print("Ответ получен:", response.text)
        print(response.json()['generated_text'])
        if response.status_code == 200:
            generation_result = response.json()['generated_text']
            return generation_result
            #write_generation_result(generation_result)
        else:
            st.error("Error 😕")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error("Error 😕")
        return None
    
        
def write_generation_result(prediction_result):
    global INPUT_TEXT_RES
    st.write("\n#### Result")
    with st.container(height=200, border=True):
        INPUT_TEXT_RES = st.write(prediction_result)
        
        
    return prediction_result
    
def get_tts_result_audio(input_text: str):
    input_features = {"input_text": input_text}
    url = f"{FASTAPI_URL}/generate_tts/"
    response = requests.get(url, params=input_features)
    if response.status_code == 200:
        generation_result = response.json()['generated_tts']
        #write_generation_result(generation_result)
    else:
        st.error("Error 😕")
        return None
    
    print(type(generation_result))
    with open("data/tts_audio.wav", "wb") as audio_file:
        audio_file.write(base64.b64decode(generation_result))
    st.audio('data/tts_audio.wav')


def get_chastushka_audio(input_text: str):
    input_features = {"input_text": input_text}
    url = f"{FASTAPI_URL}/generate_chastushka/"
    response = requests.get(url, params=input_features)
    if response.status_code == 200:
        generation_result = response.json()['generated_chastushka']
        #write_generation_result(generation_result)
    else:
        st.error("Error 😕")
        return None
    
    print(type(generation_result))
    with open("data/chast_audio.wav", "wb") as audio_file:
        audio_file.write(base64.b64decode(generation_result))
    st.audio('data/chast_audio.wav')

def generate_suno_audio(text, genre):
    URL = "https://suno-api-ts1k.vercel.app/api/custom_generate"
    params = {"prompt": text,
    "tags": genre,
    "title": "generated_song",
    "make_instrumental": False,
    "wait_audio": False}
    # Преобразование словаря в JSON
    json_data = json.dumps(params)
    # Установка заголовка Content-Type как application/json
    headers = {'Content-Type': 'application/json'}
    response = requests.post(URL, data=json_data, headers=headers)
    
    time.sleep(10)
    URL = "https://suno-api-ts1k.vercel.app/api/get"
    params = {"ids": response.json()[0]['id']}
    # Преобразование словаря в JSON
    json_data = json.dumps(params)

    # Установка заголовка Content-Type как application/json
    headers = {'Content-Type': 'application/json'}
    response = requests.get(URL, params=params)
    
    audio_bytes = requests.get(response.json()[0]['audio_url']).content
    # Отображаем аудио в виджете
    st.write("Generated song:")
    st.audio(audio_bytes, format='audio/wav')
    

if __name__ == "__main__":
    process_main_page()
    
