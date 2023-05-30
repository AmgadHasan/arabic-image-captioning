"""
A fast api that receives and image and returns the caption generated for this image
"""

from fastapi import FastAPI, Request, Form, File, UploadFile
import uvicorn
import os
import json
# import tensorflow as tf
# from tf.keras.saving import load_model
# from tf.keras.applications.inception_v3 import preprocess_input
from pydantic import BaseModel
from dummy_model import ImageLoader, ImageCaptioner#, load_tokenizer    # To be modified
from fastapi.templating import Jinja2Templates

# Declaring our FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Setting paths and directories to models
#CWD = os.getcwd()
#PWD = os.path.dirname(CWD)
#MODELS_FOLDER_PATH = PWD + "/models/"
#CNN_PATH = models_folder_path + 'cnn'
#CNN_ENCODER_PATH = MODELS_FOLDER_PATH + 'cnn_encoder'
#RNN_DECODER_PATH = MODELS_FOLDER_PATH + 'rnn_decoder'
#TOKENIZER_PATH = MODELS_FOLDER_PATH + 'tokenizer.json'

# Loading models and tokenizer
#cnn = load_model(CNN_PATH)
#cnn_encoder = load_model(CNN_ENCODER_PATH)
#rnn_decoer = load_model(RNN_DECODER_PATH)
#tokenizer = load_tokenizer(TOKENIZER_PATH)
preprocess_input = 35
dummy = 2
# Creating an image caption generator model from submodels
image_captioner = ImageCaptioner(dummy)
image_loader = ImageLoader(preprocessor = preprocess_input)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    # Use the contents of the uploadedfile to generate text for the image
    
    image = image_loader.load_image(contents)
    tokens, _ = image_captioner.predict(image)
    words = ["HI", 'hello', 'hey', 'hesy', 'asfasf','asfasfasf','asfaasfasf','..','...', 'qfqfqf','adfafd']
    caption = [words[i] for i in tokens]
    #tokens, attention_weights = custom_model.predict(image)
    #caption = tokenizer.sequences_to_texts([tokens])
    return templates.TemplateResponse("index.html", {"request": request, "prediction": caption})
