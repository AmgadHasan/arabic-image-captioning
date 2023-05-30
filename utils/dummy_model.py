# import tensorflow as tf
from constants import MAX_LENGTH, IMAGE_SIZE
import random
from PIL import Image
import io
    


class ImageCaptioner():
    """
    A custom class that builds the full model from the smaller sub models. It contains a cnn for feature extraction, a cnn_encoder to encode the features to a suitable dimension,
    an RNN decoder that contains an attention layer and RNN layer to generate text from the last predicted token + encoded image features.
    """
    def __init__(self, dummy=None, **kwargs):
        """
        Initializes the ImageCaptioner class with the given arguments.

        Args:
        cnn: A convolutional neural network that is used to extract features from images.
        cnn_encoder: A model that encodes the image features into a lower-dimensional space.
        rnn_decoder: A recurrent neural network that generates captions for the input images.
        max_length: The maximum length of the captions that the model generates.
        **kwargs: Additional keyword arguments that are not used in this implementation.
        """
        self.dummy = dummy
        self.MAX_LENGTH = MAX_LENGTH
        self.START_TOKEN_INDEX = 1
        self.END_TOKEN_INDEX = 2

    def __call__(self, inputs=None):
        """
        Calls the MyCustomModel instance with the given inputs.

        Args:
        inputs: A list of input tensors containing the decoder input, encoded features, and hidden state.

        Returns:
        The output tensor of the RNN decoder.
        """
        words = list(range(10))
        return random.choice(words)

    def predict(self, image):
        """
        Generates a caption for the given image.

        Args:
        image: An input image tensor that the model generates a caption for.

        Returns:
        A tuple containing the indices of the predicted tokens and the attention weights sequence.
        """


        caption_probability = 1
        predicted_tokens_indices = []
        attention_weights_sequence = []
        
        # Generate the caption token by token
        for i in range(self.MAX_LENGTH):

            predicted_token_index =  self.__call__(None)
            attention_weights = [2,3,4,5,6]
            predicted_tokens_indices.append(predicted_token_index)
            attention_weights_sequence.append(attention_weights)
            if predicted_token_index == self.END_TOKEN_INDEX:
                break
        
        return predicted_tokens_indices, attention_weights_sequence

    
class ImageLoader:
    """
    A helper class that loads an image from path and perform preprocessing.

    Args:
    preprocessor: A function that preprocesses input images.
    """
    def __init__(self, preprocessor=None):
        """
        Initializes the ImagePreprocessor class with the given preprocessor function.

        Args:
        preprocessor: A function that preprocesses input images.
        """
        self.preprocessor = preprocessor

    def load_image(self, path):
        """
        Loads an image from the given path, decodes it, and preprocesses it using the preprocessor function.

        Args:
        path: A path to the image file.

        Returns:
        A tuple containing the preprocessed image and the path to the original image file.
        """
        #path = path.encode('utf-8')
        image = Image.open(io.BytesIO(path))
        #print(path)
        #image = Image.open(path)
        #path = path.replace('\0', '')
        #image = Image.open(path)# Make it a batch of one image (required by tf models). Image shape is now (1, h, w, 3)
                   
        return image
