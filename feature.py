import numpy as np

from data import load_dictionary

dictionary = {}
load_dictionary()


class FeatureVector:
    entity = None
    vector = None

    def generate_vector(self, entity):
        pass


class WordFeatureVector(FeatureVector):

    def __init__(self, entity):
        word = entity.word
        self.vector = np.zeros(len(dictionary))
        self.vector[dictionary[word]] = 1
        self.entity = entity
