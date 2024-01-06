"""
A fast api that receives and image and returns the caption generated for this image
"""

from fastapi import FastAPI, Request, Form, File, UploadFile
import uvicorn
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from pydantic import BaseModel
from utils import ImageCaptioner, handle_uploaded_file
from fastapi.templating import Jinja2Templates
import argparse
import logging
import io

print("Start of script")
# Declaring our FastAPI instance and html templates
app = FastAPI()
TEMPLATES_PATH = 'utils/templates'
templates = Jinja2Templates(directory=TEMPLATES_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('--host', metavar='h', type=str, help='The host for running the webapp', default="0.0.0.0")
    parser.add_argument('-p', '--port', type=int, help='The port for running the webapp', default=8080)
    parser.add_argument('-r', '--reload', type=bool, help='Whether to run the app in reload mode', default=False)
    parser.add_argument('-l', '--log-level', help='sets the logging level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info')
    parser.add_argument('-m', '--model', type=str, help='Path to the model checkpoints', default="models/")
    parser.add_argument('-t', '--tokenizer',type=str, help='Path to the tokenizer json file', default="models/tokenizer.json")
    
    
    args = parser.parse_args()

    return args

def setup_logging(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')


@app.get("/")
async def home(request: Request):
    #return {"request": 'hello'}
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Use the contents of the uploadedfile to generate text for the image
    image_file = await handle_uploaded_file(file)
    # Generate caption text and cache attenstion weights
    caption_text, attention_weights = image_captioner.predict(image_file)
    
    return  {"caption": caption_text, "attention_weights": attention_weights}

if __name__ == '__main__':
    args = parse_args()

    logging.info(f"Running web app with log level: {args.log_level}")
    setup_logging(args.log_level)

    image_captioner = ImageCaptioner(model_path=args.model, tokenize_path=args.tokenizer, preprocessor=preprocess_input)

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
