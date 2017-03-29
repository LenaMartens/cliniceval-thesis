import os

import utils
from feature import FeatureVector

import gensim


class WordEmbedding(FeatureVector):
    # shared by all instances
    model = None

    def __init__(self, entity, filepath):
        super().__init__(entity)
        if self.model is None:
            self.model = train_model(filepath)

    def generate_vector(self):
        return self.model[self.entity.word.lower()]


def train_model(filepath):
    sentences = utils.sentence_generator(filepath)
    return gensim.models.Word2Vec(sentences)
