import tensorflow as tf
from constants import MAX_LENGTH, IMAGE_SIZE
import json
from PIL import Image
import io


def load_tokenizer(file_path):
    """A helper function to load tokenizer saved as json file."""
    with open(file_path) as file:
        data = json.load(file)
        loaded_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
        return loaded_tokenizer


class ImageCaptioner():
    """
    A custom class that builds the full model from the smaller sub models. It contains a cnn for feature extraction, a cnn_encoder to encode the features to a suitable dimension,
    an RNN decoder that contains an attention layer and RNN layer to generate text from the last predicted token + encoded image features.
    """
    def __init__(self, cnn, cnn_encoder, rnn_decoder, **kwargs):
        """
        Initializes the ImageCaptioner class with the given arguments.

        Args:
        cnn: A convolutional neural network that is used to extract features from images.
        cnn_encoder: A model that encodes the image features into a lower-dimensional space.
        rnn_decoder: A recurrent neural network that generates captions for the input images.
        max_length: The maximum length of the captions that the model generates.
        **kwargs: Additional keyword arguments that are not used in this implementation.
        """
        self.cnn = cnn
        self.cnn_encoder = cnn_encoder
        self.rnn_decoder = rnn_decoder
        self.MAX_LENGTH = MAX_LENGTH
        self.START_TOKEN_INDEX = 1
        self.END_TOKEN_INDEX = 2

    def __call__(self, inputs):
        """
        Calls the MyCustomModel instance with the given inputs.

        Args:
        inputs: A list of input tensors containing the decoder input, encoded features, and hidden state.

        Returns:
        The output tensor of the RNN decoder.
        """
        [decoder_input, encoded_features, hidden_state] = inputs
        return self.rnn_decoder(decoder_input, encoded_features, hidden_state, training=False)

    def predict(self, image):
        """
        Generates a caption for the given image.

        Args:
        image: An input image tensor that the model generates a caption for.

        Returns:
        A tuple containing the indices of the predicted tokens and the attention weights sequence.
        """
        image_features = self.cnn(image)
        reshaped_features = tf.reshape(image_features, (tf.shape(image_features)[0], -1, image_features.shape[3]))
        encoded_features = self.cnn_encoder(reshaped_features)

        # Get the RNN's initial state and start token for each new sample
        hidden_state = tf.zeros((1, 512))
        decoder_input = tf.expand_dims([self.START_TOKEN_INDEX],0)
        decoder_input = tf.cast(decoder_input, tf.int32)
        caption_probability = 1
        predicted_tokens_indices = []
        attention_weights_sequence = []
        
        # Generate the caption token by token
        for i in range(self.MAX_LENGTH):
            logits, hidden_state, attention_weights = self.__call__([decoder_input, encoded_features, hidden_state])
            predicted_token_index = tf.cast(tf.random.categorical(logits, 1)[0][0], tf.int64)
            predicted_tokens_indices.append(tf.get_static_value(predicted_token_index))
            attention_weights_sequence.append(attention_weights)
            if predicted_token_index == self.END_TOKEN_INDEX:
                break
            decoder_input = tf.expand_dims([tf.cast(predicted_token_index, tf.int32)], 0)
        
        return predicted_tokens_indices, attention_weights_sequence

    
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
        self.preprocessor = preprocessor

    def load_image(self, path):
        """
        Loads an image from the given path, decodes it, and preprocesses it using the preprocessor function.

        Args:
        path: A path to the image file.

        Returns:
        A tuple containing the preprocessed image and the path to the original image file.
        """
        image = Image.open(io.BytesIO(path))
        image = tf.image.resize(image, IMAGE_SIZE)
        image = self.preprocessor(image)
        image = tf.expand_dims(image, 0)    # Make it a batch of one image (required by tf models). Image shape is now (1, h, w, 3)
        return image
