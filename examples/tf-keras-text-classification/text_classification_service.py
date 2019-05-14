import pandas as pd
import numpy as np
import tensorflow as tf
from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TfKerasModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

@artifacts([
    TfKerasModelArtifact('model'),
    PickleArtifact('word_index')
])
@env(conda_dependencies=['tensorflow', 'numpy', 'pandas'])
class TextClassificationService(BentoService):
   
    def word_to_index(self, word):
        if word in self.artifacts.word_index:
            return self.artifacts.word_index[word]
        else:
            return self.artifacts.word_index["<UNK>"]
    
    @api(JsonHandler)
    def predict(self, parsed_json):
        """
        """
        text = parsed_json['text']
        
        sequence = tf.keras.preprocessing.text.hashing_trick(
            text,
            256,
            hash_function=self.word_to_index,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ')
        
        with tf.get_default_graph().as_default():
            with tf.Session().as_default():
                return self.artifacts.model.predict(np.expand_dims(sequence, 0))
