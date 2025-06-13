import json
from tensorflow import keras
import tensorflow as tf

from llm_model import fast_label_smoothing_loss
from env_config import *


class LlmModelEntity:

    @staticmethod
    def save(model, model_id):
        """
        Saves a Keras model to pb file
        """
        model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"
        model.save(model_pathname)

        print(f"\nThe model has been saved to {model_pathname}")

    @staticmethod
    def save_metadata(metadata, model_id):
        """
        Saves Keras model's metadata to JSON file
        """
        metadata_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}/{MODEL_METADATA_FILENAME}"
        with open(metadata_pathname, 'w') as f:
            json.dump(metadata, f, indent=2)

            print(f"The model metadata has been saved to {metadata_pathname}")

    @staticmethod
    def save_evaluation(model_id, data):
        """
        Saves model's evaluation data to JSON
        """
        metadata_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}/{MODEL_EVAL_DATA_FILENAME}"
        with open(metadata_pathname, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nThe model evaluation data {data} has been saved to {metadata_pathname}")

    @staticmethod
    def load(model_id):
        """
        Loads model
        """
        model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"

        model = tf.keras.models.load_model(
            model_pathname,
            custom_objects={
                'fast_label_smoothing_loss': fast_label_smoothing_loss
            })

        print(f"Model {model_id} has been loaded")
        return model

    @staticmethod
    def load_metadata(model_id):
        """
        Loads the model's metadata
        """
        with open(f"{MODELS_DIRECTORY_PATH}/{model_id}/{MODEL_METADATA_FILENAME}", 'r') as f:
            metadata = json.load(f)

            print(f"The Model's metadata has been loaded\n{metadata}\n")

        return metadata
