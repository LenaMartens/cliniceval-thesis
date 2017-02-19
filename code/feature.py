import nltk
from nltk.data import load
import numpy as np
import utils

dictionary = utils.get_dictionary()
'''
A vector denoting a single feature of a word
'''


class FeatureVector:
    def get_vector(self):
        return self.vector

    def generate_vector(self):
        self.vector = []

    def __init__(self, entity):
        self.entity = entity
        self.vector = []
        self.generate_vector()


'''
A vector denoting a single feature of a relationship between two phrases
'''


class RelationFeatureVector(FeatureVector):
    def __init__(self, source, target):
        self.source = source
        self.vector = []
        self.target = target
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


'''
Is the word capitalised?
'''


class CapitalFeatureVector(FeatureVector):
    def generate_vector(self):
        self.vector = [self.entity.word.title()]

'''
What DOCTIME has the word been classified as?
'''
doctimes = utils.get_doctimes()

class DocTimeVector(FeatureVector):
    def generate_vector(self):
        self.vector = np.zeros(len(doctimes)+1)
        try:
            self.vector[doctimes[self.entity.word.doc_time_rel]] = 1
        except KeyError:
            self.vector[len(dictionary)] = 1
tagdict = load('help/tagsets/upenn_tagset.pickle')
tag_list = list(tagdict.keys())

'''
One hot encoding of part of speech tag.
'''


class POSFeatureVector(FeatureVector):
    def generate_vector(self):
        [(word, tag)] = nltk.pos_tag([self.entity.word])
        self.vector = np.zeros(len(tag_list))
        self.vector[tag_list.index(tag)] = 1


'''
Do the two entities appear in the same paragraph?
'''


class SameParVector(RelationFeatureVector):
    def generate_vector(self):
        self.vector = [self.source.paragraph == self.target.paragraph]


'''
Specific feature vectors used in training and prediction
'''


class WordVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordFeatureVector(self.entity))
        self.features.append(POSFeatureVector(self.entity))
        self.features.append(CapitalFeatureVector(self.entity))
        self.features.append(DocTimeVector(self.entity))


class TimeRelationVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordVector(self.entity.source))
        self.features.append(WordVector(self.entity.target))
        self.features.append(SameParVector(self.entity.source, self.entity.target))
