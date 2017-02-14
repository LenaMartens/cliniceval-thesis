import nltk
from nltk.data import load
import numpy as np
import utils

dictionary = utils.get_dictionary()

'''
A vector denoting a single feature
'''


class FeatureVector:
    def get_vector(self):
        return self.vector

    def generate_vector(self):
        self.vector = []

    def __init__(self, entity):
        self.entity = entity
        self.generate_vector()


'''
A vector consisting of several features. Is implemented as a list of FeatureVectors
'''


class ConcatenatedVector:
    def get_vector(self):
        return np.concatenate([x.get_vector() for x in self.features])

    def __init__(self, entity):
        self.entity = entity
        self.features = list()
        self.generate_vector()

    def generate_vector(self):
        pass


'''
One hot encoding of the word in the dictionary of encountered words.
'''


class WordFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(dictionary) + 1)
        try:
            self.vector[dictionary[word]] = 1
        except KeyError:
            self.vector[len(dictionary)] = 1


tagdict = load('help/tagsets/upenn_tagset.pickle')
tag_list = list(tagdict.keys())

'''
One hot encoding of part of speech tag.
'''


class POSFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        [(word, tag)] = nltk.pos_tag([word])
        self.vector = np.zeros(len(tag_list))
        self.vector[tag_list.index(tag)] = 1


class RelationFeatureVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordFeatureVector(self.entity.source))
        self.features.append(POSFeatureVector(self.entity.source))
        self.features.append(WordFeatureVector(self.entity.target))
        self.features.append(POSFeatureVector(self.entity.target))




