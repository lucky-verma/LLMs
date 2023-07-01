# By: Lucky Verma
# Date: 30th June 2023

# Import -----------------------------------
import os
import sys
import time
import requests

import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from PIL import Image

# Load model and processor -------------------
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ST Configuration --------------------------------
st.set_page_config(
    page_title="BLIP-2",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# hide hamburger menu and footer
hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Page layout -----------------------------------
st.title("BLIP-2")

st.sidebar.title("Tuning Parameters")

if 'image' not in st.session_state:
    st.session_state['image'] = None

st.write("Upload the input Image or input the URL of the image")

# select option to upload image or input url
option = st.selectbox(
    "Select the option",
    ("Input URL", "Upload Image"),
)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state['image'] = image

elif option == "Input URL":
    url = st.text_input("Enter the URL of the image", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png")
    if url is not None:
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        st.session_state['image'] = image

# if image is not uploaded or url is not given
image = st.session_state['image']
if image is None:
    st.write("Please upload the image or input the URL of the image")
    sys.exit()

st.image(image, caption="Uploaded Image", use_column_width=True)

# select the tasks (Image captioning, Prompted image captioning, Visual question answering (VQA), Chat-based prompting)
tasks = st.sidebar.multiselect(
    "Select the tasks",
    ("Image captioning", "Prompted image captioning", "Visual question answering (VQA)", "Chat-based prompting"),
)

# image captioning
if "Image captioning" in tasks:
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.write("Generated caption: ", generated_text)

# prompted image captioning
if "Prompted image captioning" in tasks:
    prompt = st.text_input("Enter the prompt", "This is a picture of")
    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.write("Generated caption: ", generated_text)

# visual question answering (VQA)
if "Visual question answering (VQA)" in tasks:
    question = st.text_input("Enter the question", "Question: which city is this? Answer:")
    inputs = processor(question, image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.write("Answer: ", generated_text)

# chat-based prompting
if "Chat-based prompting" in tasks:
    context = [
        ("which city is this?", "singapore"),
        ("why?", "it has a statue of a merlion"),
    ]
    question = "where is the name merlion coming from?"
    template = "Question: {} Answer: {}."

    prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"

    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.write("Answer: ", generated_text)

