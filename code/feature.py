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
    def get_vector(self):
        return self.vector

    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(dictionary) + 1)
        try:
            self.vector[dictionary[word]] = 1
        except KeyError:
            self.vector[len(dictionary)] = 1


class RelationFeatureVector(FeatureVector):
    def get_vector(self):
        return np.concatenate([x.get_vector() for x in self.features])

    def generate_vector(self):
        self.features = list()
        self.features.append(WordFeatureVector(self.entity.source))
        self.features.append(WordFeatureVector(self.entity.target))
