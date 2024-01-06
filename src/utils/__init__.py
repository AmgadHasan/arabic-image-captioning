import io
import pathlib
import json
from fastapi import UploadFile

def load_config(path: pathlib.Path) -> pathlib.Path:
    """
    A helper function to load a JSON configuration file as a python dictionary.
    
    Args:
    path (pathlib.Path): The path to the JSON configuration file.
    
    Returns:
    dict: The loaded configuration data as a Python dictionary.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config

async def handle_uploaded_file(uploaded_file: UploadFile):
    """
    A helper function to handle uploaded files.

    This function takes an UploadFile object as an argument and returns an
    io.BytesIO object containing the file's contents. The function reads the
    uploaded file's contents and creates a BytesIO object to represent the
    file's data in memory.

    Args:
    uploaded_file (UploadFile): The uploaded file to be processed.

    Returns:
    io.BytesIO: An in-memory representation of the uploaded file's contents.
    """
    contents = await uploaded_file.read()
    image_file = io.BytesIO(contents)

    return image_file
