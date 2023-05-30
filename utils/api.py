"""
A fast api that receives and image and returns the caption generated for this image
"""

from fastapi import FastAPI, Request, Form, File, UploadFile
import uvicorn
import os
import io
import json
import tensorflow as tf
from tensorflow.keras.saving import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from pydantic import BaseModel
from full_model import ImageLoader, ImageCaptioner, load_tokenizer    # To be modified
from fastapi.templating import Jinja2Templates

# Declaring our FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Setting paths and directories to models
CWD = os.getcwd()
PWD = os.path.dirname(CWD)
MODELS_FOLDER_PATH = PWD + "/models/"
CNN_PATH = MODELS_FOLDER_PATH + 'cnn'
CNN_ENCODER_PATH = MODELS_FOLDER_PATH + 'cnn_encoder'
RNN_DECODER_PATH = MODELS_FOLDER_PATH + 'decoder'
TOKENIZER_PATH = MODELS_FOLDER_PATH + 'tokenizer.json'

# Loading models and tokenizer
cnn = load_model(CNN_PATH)
cnn_encoder = load_model(CNN_ENCODER_PATH)
rnn_decoer = load_model(RNN_DECODER_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)


# Creating an image caption generator model from submodels
image_captioner = ImageCaptioner(cnn, cnn_encoder, rnn_decoer)
image_loader = ImageLoader(preprocessor = preprocess_input)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    
    # Use the contents of the uploadedfile to generate text for the image
    #contents = await file.read()
    # Open the buffer as an image using the Python Imaging Library (PIL)
    contents = await file.read()
    # Use the contents of the uploadedfile to generate text for the image
    
    image = image_loader.load_image(contents)
    # Load and preprocess the image using the custom class
    #image = image_loader.load_image(buffer)
    # Generate tokens and cache attenstion weights
    tokens, attention_weights = image_captioner.predict(image)
    # Convert token idexes to words
    caption = tokenizer.sequences_to_texts([tokens])
    return templates.TemplateResponse("index.html", {"request": request, "caption": caption})
