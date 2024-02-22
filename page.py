# python native
import os 
import json
import shutil
import random
import datetime
import argparse

# external library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict

# torch
import torch 

# streamlit
import streamlit as st

# diffusion
from lora_diffusion import tune_lora_scale, patch_pipe
from diffusers import StableDiffusionImg2ImgPipeline

# custom module
from split_image import *


def run(**kargs):
    cfg = EasyDict(kargs)
    
    # Init pretrained pipeline
    with st.spinner('Wait for it...'):
        pipe = get_sd_pipeline(cfg)
    
    
    # Init variables
    if "patch" not in st.session_state:
        st.session_state['patch'] = None
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "files_uploader_key" not in st.session_state:
        st.session_state["files_uploader_key"] = 10000


    # Title
    draw_title("Background Asset Generator")
    
    
    # Sidebar
    draw_side_header("Select Pretrained Model")  
      
    last_selected_row = draw_pretrained_list()
    if last_selected_row is not None:
        model_name = str(last_selected_row).split()[1]
        st.sidebar.write("Selected model:", model_name)
        patch(cfg, pipe, model_name)
    
        if st.sidebar.button("Delete selected model"):
            shutil.rmtree(os.path.join(cfg.MODEL_DIR, model_name))
            st.rerun()
    
    # Main - Inference
    draw_header("Inference")
    checkbox = draw_checkbox("I'll customize hyper parameters")        

    draw_subheader("Upload Content Image")
    uploaded_image = upload_image()
    if st.button('Clear uploaded content image'):
        st.session_state.file_uploader_key += 1
        st.rerun()
    
    if checkbox:
        draw_subheader("Set Hyper-parameters")
        alpha_unet = st.slider('alpha_unet', 0.3, 1., 0.3, step=0.1)
        alpha_text = st.slider('alpha_text', 0.3, 1., 0.3, step=0.1)
        strength = st.slider('strength', 0., 1., 0.5, step=0.05)
        guidance_scale = st.slider('guidance_scale', 1., 15., 7.5, step=0.5)
            
        if st.button('Start inference'):
            if uploaded_image:
                draw_subheader("Result")
                with st.spinner('Wait for it...'):
                    style_transfer(cfg, pipe, uploaded_image,
                            alpha_unet=alpha_unet, alpha_text=alpha_text,
                            strength=strength, guidance_scale=guidance_scale)
    else:
        alpha_unets = [0.3, 0.45, 0.6]
        alpha_texts = [0.3, 0.6, 0.8]
        strengths = [0.45, 0.55, 0.65, 0.75]
        
        if st.button('Start inference'):
            if uploaded_image:
                draw_subheader("Result")
                for strength in strengths:
                    with st.spinner('Wait for it...'):
                        style_transfer_table(cfg, pipe, uploaded_image,
                            alpha_unets=alpha_unets, alpha_texts=alpha_texts,
                            strength=strength) 
                        
            
    # Main - Train
    draw_header("Train")
    draw_subheader("Upload Style Images")
    col1, col2 = st.columns(2)
    with col1:
        train_model_name = st.text_input('Train model name', 'temp')
        cfg['TRAIN_MODEL_NAME'] = train_model_name
    with col2:
        rank = st.number_input("LoRA rank", 2)
        cfg['RANK'] = rank
    st.write(f"Trained model will be named `{train_model_name}_r{rank}`")
        
    uploaded_images = upload_images()
    if st.button('Clear uploaded style images'):
        st.session_state.files_uploader_key += 1
        st.rerun()
            
    if st.button('Start training'):
        if uploaded_images:
            with st.spinner('Saving images...'):
                remove_and_save_images(cfg, uploaded_images)
        cmd = get_train_command(cfg)
        with st.spinner('Now Training... It takes about 10 minutes.'):
            train(cmd)
    
        st.balloons()
        st.rerun()
    
    
def draw_title(txt): st.title(txt)
def draw_header(txt): st.header(txt)
def draw_subheader(txt): st.subheader(txt)
def draw_side_header(txt): st.sidebar.header(txt)
def draw_checkbox(txt): return st.checkbox(txt)
def draw_pretrained_list():   
    
    def get_data():
        lst = os.listdir('exps')
        return pd.DataFrame(lst, columns=['model_name'])


    df = get_data()
    df["select"] = False
    st.session_state["data"] = df

    if "editor_key" not in st.session_state:
        st.session_state["editor_key"] = random.randint(0, 100000)

    if "last_selected_row" not in st.session_state:
        st.session_state["last_selected_row"] = None


    def get_row_and_clear_selection():
        key = st.session_state["editor_key"]
        df = st.session_state["data"]
        selected_rows = st.session_state[key]["edited_rows"]
        selected_rows = [int(row) for row in selected_rows if selected_rows[row]["select"]]
        try:
            last_row = selected_rows[-1]
        except IndexError:
            return
        df["select"] = False
        st.session_state["data"] = df
        st.session_state["editor_key"] = random.randint(0, 100000)
        st.session_state["last_selected_row"] = df.iloc[last_row]


    st.sidebar.data_editor(
        st.session_state["data"],
        key=st.session_state["editor_key"],
        on_change=get_row_and_clear_selection,
    )

    row = None
    if "last_selected_row" in st.session_state:
        row = st.session_state["last_selected_row"]
        
    return row

def upload_image():
    uploaded_file = st.file_uploader(
        "Upload an image.", 
        type=["png", "jpg", "jpeg"],
        key=st.session_state["file_uploader_key"],
    )
    
    return uploaded_file

def upload_images():
    uploaded_file = st.file_uploader(
        "Upload an image.", 
        type=["png", "jpg", "jpeg"],
        key=st.session_state["files_uploader_key"],
        accept_multiple_files=True
    )
    
    return uploaded_file

def remove_and_save_images(cfg, uploaded_images):
    try:
        shutil.rmtree(os.path.join(cfg.DATA_DIR, cfg.TRAIN_MODEL_NAME))
    except: pass
    os.makedirs(os.path.join(cfg.DATA_DIR, cfg.TRAIN_MODEL_NAME), exist_ok=True)
    for upload_image in uploaded_images:
        image_name = upload_image.name
        pil_image = Image.open(upload_image)
        if cfg.WITH_PATCH:
            np_image = pil_to_numpy(pil_image)
            np_patches = divide_image_into_patches(np_image)
            for i, np_patch in enumerate(np_patches):
                pil_patch = numpy_to_pil(np_patch)
                pil_patch.save(os.path.join(cfg.DATA_DIR, cfg.TRAIN_MODEL_NAME, f"{i}_{image_name.lower()}"))
        else:
            if cfg.GRAY: pil_image = pil_image.convert("L")
            pil_image.save(os.path.join(cfg.DATA_DIR, cfg.TRAIN_MODEL_NAME, image_name.lower()))

def style_transfer_table(cfg, pipe, uploaded_image,
                   alpha_unets, alpha_texts,
                   strength=0.55, guidance_scale=7.5):

    if cfg.GRAY: 
        init_image = Image.open(uploaded_image).convert("L").convert("RGB").resize((512, 512))
    else:
        init_image = Image.open(uploaded_image).convert("RGB").resize((512, 512))
            
    fig = plt.figure(figsize=(20,20)) # Notice the equal aspect ratio
    ax = [fig.add_subplot(len(alpha_unets),len(alpha_texts),i+1) for i in range(len(alpha_texts)*len(alpha_unets))]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')

    fig.subplots_adjust(wspace=0, hspace=0)

    for i, alpha_unet in enumerate(alpha_unets):
        for j, alpha_text in enumerate(alpha_texts):
            tune_lora_scale(pipe.unet, alpha_unet)
            tune_lora_scale(pipe.text_encoder, alpha_text)
            torch.manual_seed(cfg.SEED) # 동일 조건 동일 결과 보장
            image = pipe(prompt="style of <s1><s2>", 
                        image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]

            ax[i*len(alpha_texts) + j].imshow(image)   
    
    st.write(f"strength: {strength}, alpha_unets: {alpha_unets}, alpha_texts: {alpha_texts}")
    fig
            
def style_transfer(cfg, pipe, uploaded_image,
                   alpha_unet=0.4, alpha_text=0.8,
                   strength=0.55, guidance_scale=7.5):
    
    if cfg.GRAY: 
        init_image = Image.open(uploaded_image).convert("L").convert("RGB").resize((512, 512))
    else:
        init_image = Image.open(uploaded_image).convert("RGB").resize((512, 512))

    tune_lora_scale(pipe.unet, alpha_unet)
    tune_lora_scale(pipe.text_encoder, alpha_text)
    torch.manual_seed(cfg.SEED) # 동일 조건 동일 결과 보장
    
    image = pipe(prompt="style of <s1><s2>", 
                image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]
    
    fig = plt.figure(figsize=(16,8)) # Notice the equal aspect ratio
    ax = [fig.add_subplot(2,1,i+1) for i in range(2)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].imshow(image)
    ax[1].imshow(init_image)
    
    fig

def patch(cfg, pipe, model_name):
    if st.session_state['patch'] == model_name: return 
    st.session_state['patch'] = model_name
    print("PATCH", model_name)

    patch_pipe(
        pipe,
        os.path.join(cfg.MODEL_DIR, model_name, "final_lora.safetensors"),
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

# @st.cache_data          # NOTE: 디버깅 시 사용합니다.
@st.cache_resource    # NOTE: 배포 시 사용합니다.
def get_sd_pipeline(cfg):
    model_id = cfg.MODEL_NAME
    return StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda"
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(command):
    os.system(command)
    get_sd_pipeline.clear() # st.cache_resource clear

def get_train_command(cfg):
    return f"""lora_pti \
  --pretrained_model_name_or_path={cfg.MODEL_NAME}  \
  --instance_data_dir={os.path.join(cfg.DATA_DIR, cfg.TRAIN_MODEL_NAME)} \
  --output_dir={os.path.join(cfg.MODEL_DIR, cfg.TRAIN_MODEL_NAME+"_r"+str(cfg.RANK))} \
  --train_text_encoder \
  --resolution={cfg.RESOLUTION} \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet={cfg.LR_UNET} \
  --learning_rate_text={cfg.LR_TEXT} \
  --learning_rate_ti={cfg.LR_TI} \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="style"\
  --save_steps=100 \
  --max_train_steps_ti={cfg.STEP_TI} \
  --max_train_steps_tuning={cfg.STEP_TUNING} \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank={cfg.RANK} \
  --seed={cfg.SEED}"""
  


st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #dbcca0;
        }
    </style>
    """, unsafe_allow_html=True)

run(
    MODEL_NAME="runwayml/stable-diffusion-v1-5",
    SEED="42",
    DATA_DIR="./data",  # edit for train
    MODEL_DIR="./exps", # edit for inference
    RESOLUTION=512,     # edit for train
    LR_UNET=1e-4,       # edit for train
    LR_TEXT=1e-5,       # edit for train
    LR_TI=5e-4,         # edit for train
    STEP_TI=500,       # edit for train
    STEP_TUNING=500,   # edit for train
    WITH_PATCH=False,
    GRAY=True,
)