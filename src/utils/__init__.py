import tensorflow as tf
import json
from PIL import Image
import io
from .full_model import ImageCaptioner
from .constants import IMAGE_SIZE


class ImageLoader:
    """
    A helper class that loads an image from path and perform preprocessing.

    Args:
    preprocessor: A function that preprocesses input images.
    """
    def __init__(self, preprocessor):
        """
        Initializes the ImagePreprocessor class with the given preprocessor function.

        Args:
        preprocessor: A function that preprocesses input images.
        """
        

    def load_image(self, path):
        """
        Loads an image from the given path, decodes it, and preprocesses it using the preprocessor function.

        Args:
        path: A path to the image file.

        Returns:
        A tuple containing the preprocessed image and the path to the original image file.
        """
        image = Image.open(path)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = self.preprocessor(image)
        image = tf.expand_dims(image, 0)    # Make it a batch of one image (required by tf models). Image shape is now (1, h, w, 3)
        return image
