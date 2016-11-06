import numpy as np
import utils

dictionary = utils.get_dictionary()


class FeatureVector:
    def generate_vector(self):
        pass

    def __init__(self, entity):
        self.entity = entity
        self.generate_vector()


class WordFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(dictionary))
        self.vector[dictionary[word]] = 1


class RelationFeatureVector(FeatureVector):
    def generate_vector(self):
        word1 = self.entity.source.word
        word2 = self.entity.target.word
        self.vector = np.zeros(len(dictionary) * 2)
        self.vector[dictionary[word1]] = 1
        self.vector[len(dictionary) + dictionary[word2] - 1] = 1
