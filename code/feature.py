import nltk
from nltk.data import load
import numpy as np
import utils

dictionary = utils.get_dictionary()


class FeatureVector:
    def get_vector(self):
        return self.vector

    def generate_vector(self):
        self.vector = []

    def __init__(self, entity):
        self.entity = entity
        self.generate_vector()


class WordFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(dictionary) + 1)
        try:
            self.vector[dictionary[word]] = 1
        except KeyError:
            self.vector[len(dictionary)] = 1


tagdict = load('help/tagsets/upenn_tagset.pickle')
tag_list = tagdict.keys()


class POSFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        (word, tag) = nltk.pos_tag(word)
        self.vector = np.zeros(len(tag_list))
        self.vector[tag_list.find(tag)] = 1


class RelationFeatureVector(FeatureVector):
    def get_vector(self):
        return np.concatenate([x.get_vector() for x in self.features])

    def generate_vector(self):
        self.features = list()
        self.features.append(WordFeatureVector(self.entity.source))
        self.features.append(WordFeatureVector(self.entity.target))
