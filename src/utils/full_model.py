import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib
import json


def load_config(path: pathlib.Path) -> pathlib.Path:
    """
    A helper function to load a JSON config.

    Args:
    path (pathlib.Path): The path to the saved model.

    Returns:
    dict: The loaded config as a Python dict.
    """
    with open(path) as f:
        config = json.load(f)

    return config


class Tokenizer:
    def __init__(self, path: str):
        self.config = load_config(path / "tokenizer_config.json")
        self.tokenizer = self.load_from_json(path / "tokenizer.json")

    def load_from_json(self, file_path: pathlib.Path) -> tf.keras.preprocessing.text.Tokenizer:
        """
        A helper function to load tokenizer saved as JSON file.

        Args:
        file_path (pathlib.Path): The path to the tokenizer JSON file.

        Returns:
        tf.keras.preprocessing.text.Tokenizer: The loaded tokenizer.
        """
        with open(file_path) as file:
            data = json.load(file)
            loaded_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
        
        return loaded_tokenizer

class Model:
    def __init__(self, path: str):
        self.config = load_config(path / "model_config.json")
        self.cnn = self._load_model(path / "cnn")
        self.cnn_projector = self._load_model(path / "cnn_projector")
        self.rnn_decoder = self._load_model(path / "decoder")

    def _load_model(self, path: pathlib.Path) -> tf.keras.Model:
        """
        A helper function to load a saved Keras model from the given path.

        Args:
        path (pathlib.Path): The path to the saved model.

        Returns:
        tf.keras.Model: The loaded Keras model.
        """
        return load_model(path)

    def encode(self, images) -> tf.Tensor:
        """
        Encodes the input images and returns the encoded features.

        Args:
        images (tf.Tensor): The input images tensor.

        Returns:
        tf.Tensor: The encoded features tensor.
        """
        images_features = self.cnn(images)
        reshaped_features = tf.reshape(images_features, (tf.shape(images_features)[0], -1, images_features.shape[3]))
        encoded_features = self.cnn_projector(reshaped_features)

        return encoded_features

    def decode(self, decoder_inputs, encoded_features, hidden_states) -> dict:
        """
        Decodes the input and returns the logits, hidden states, and attention weights.

        Args:
        decoder_inputs (tf.Tensor): The decoder input tensor.
        encoded_features (tf.Tensor): The encoded features tensor.
        hidden_states (tf.Tensor): The hidden states tensor.

        Returns:
        dict: A dictionary containing the logits, hidden states, and attention weights.
        """
        logits, hidden_states, attention_weights = self.rnn_decoder([decoder_inputs, encoded_features, hidden_states])

        return {"logits": logits, "hidden_states": hidden_states, "attention_weights": attention_weights}

    def __call__(self, images, decoder_inputs, hidden_states) -> dict:
        """
        Calls the MyCustomModel instance with the given inputs.

        Args:
        images (tf.Tensor): The input images tensor.
        decoder_inputs (tf.Tensor): The decoder input tensor.
        hidden_states (tf.Tensor): The hidden states tensor.

        Returns:
        dict: A dictionary containing the logits, hidden states, and attention weights.
        """
        encoded_features = self.encode(images)
        
        outputs = self.decode(decoder_inputs, encoded_features, hidden_states)

        return outputs

 
class ImageCaptioner():
    """
    A custom class that builds the full model from the smaller sub-models. It contains a CNN for feature extraction, a CNN encoder to encode the features to a suitable dimension,
    an RNN decoder that contains an attention layer and RNN layer to generate text from the last predicted token + encoded image features.
    """
    def __init__(self, model_path: pathlib.Path, tokenizer_path, preprocessor):
        """
        Initializes the ImageCaptioner class with the given arguments.

        Args:
        path (pathlib.Path): The path to the directory containing the saved models and configuration files.
        **kwargs: Additional keyword arguments that are not used in this implementation.
        """
        self.preprocessor = preprocessor

        self.tokenizer = Tokenizer(tokenizer_path) 
        
        self.model = Model(model_path)

    def predict(self, images, max_length, num_captions=5):
        """
        Generates a caption for the given image.

        Args:
        image: An input image tensor that the model generates a caption for.

        Returns:
        A tuple containing the indices of the predicted tokens and the attention weights sequence.
        """
        if not max_length or max_length > self.model.config['max_length']:
            max_length = self.model.config['max_length']
        
        images = tf.image.resize(images, self.model.config["image_size"])
        images = self.preprocessor(images)
        
        encoded_features = self.model.encode(images)

        # Get the RNN's initial state and start token for each new sample
        # hidden_state = tf.zeros((1, 512))
        # decoder_input = tf.expand_dims([self.START_TOKEN_INDEX],0)
        # decoder_input = tf.cast(decoder_input, tf.int32)
        # caption_probability = 1
        # predicted_tokens_indices = []
        # attention_weights_sequence = []
        results = tf.Variable(tf.zeros(shape=(num_captions, max_length),dtype='int32'), )
        scores = tf.ones(shape=(num_captions,))
        #hidden = decoder.get_initial_state(batch_size=1)
        #hiddens = self.rnn_decoder.get_initial_state(batch_size=n_captions)
        hidden_states = tf.zeros((num_captions, self.model.config["num_hidden_units"]))
        dec_inputs = tf.fill(dims=(n_captions,1), value=self.tokenizer_config['bos_token_id'])
        batch_indices = list(range(n_captions)) # batch size
        for i in range(max_length):
            logits, hidden_states, attention_weights = self.model.decode(decoder_inputs, encoded_features, hidden_states)
            predicted_ids = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)  # shape (batch_size,num_samples)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)
            #predicted_ids = tf.convert_to_tensor(predicted_ids, dtype=tf.int32)#tf.cast(predicted_ids, tf.int32)
            #probabilities = tf.nn.softmax(logits, axis=-1)
            element_indices = predicted_ids
            
            indices = tf.stack([batch_indices, element_indices], axis=1)
            scores *= tf.gather_nd(logits ,indices = indices)
            #predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int64).numpy()[0]
            #print(predicted_id)
            #print(predicted_ids)
            results[:,i].assign(predicted_ids)
            
            # if tokenizer.index_word[predicted_id] == 'نه':
            #     break
            dec_inputs = tf.expand_dims(predicted_ids, 1)
            #dec_input = tf.expand_dims([predicted_id], 0)
            #print(probs)
        most_probable_sequence_id = int(tf.math.argmax(scores))
        best_caption = list(results[most_probable_sequence_id].numpy())
        print(best_caption)
        eos_loc = best_caption.index(self.tokenizer_config['eos_token_id'])
        #caption_text = tokenizer.sequences_to_texts([best_caption[:eos_loc]])
        
        return best_caption[:eos_loc], None
        # Generate the caption token by token
        # for i in range(self.MAX_LENGTH):
        #     logits, hidden_state, attention_weights = self.__call__([decoder_input, encoded_features, hidden_state])
        #     predicted_token_index = tf.cast(tf.random.categorical(logits, 1)[0][0], tf.int64)
        #     predicted_tokens_indices.append(tf.get_static_value(predicted_token_index))
        #     attention_weights_sequence.append(attention_weights)
        #     if predicted_token_index == self.END_TOKEN_INDEX:
        #         break
        #     decoder_input = tf.expand_dims([tf.cast(predicted_token_index, tf.int32)], 0)
        
        # return predicted_tokens_indices, attention_weights_sequence
