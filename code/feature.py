import nltk
from nltk.data import load
import numpy as np
import utils
import scipy
import scipy.sparse

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
        return scipy.concatenate([x.get_vector() for x in self.features])

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
        self.vector = scipy.sparse.csr_matrix(self.vector)


'''
Is the word capitalised?
'''


class CapitalFeatureVector(FeatureVector):
    def generate_vector(self):
        self.vector = [self.entity.word.istitle()]


'''
What DOCTIME has the word been classified as?
'''


class DocTimeVector(FeatureVector):
    def generate_vector(self):
        doctimes = utils.get_doctimes()
        self.vector = np.zeros(len(doctimes) + 1)
        try:
            self.vector[doctimes[self.entity.doc_time_rel]] = 1
        except KeyError:
            print(self.entity.doc_time_rel)
            self.vector[len(doctimes)] = 1
        self.vector = scipy.sparse.csr_matrix(self.vector)


tagdict = load('help/tagsets/upenn_tagset.pickle')
tag_list = list(tagdict.keys())

'''
One hot encoding of part of speech tag.
'''


class POSFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(tag_list))
        if word:
            [(word, tag)] = nltk.pos_tag([word])
            self.vector[tag_list.index(tag)] = 1
        self.vector = scipy.sparse.csr_matrix(self.vector)


'''
One hot encoding of polarity of the word (from input file)
'''


class PolarityFeatureVector(FeatureVector):
    def generate_vector(self):
        polarities = utils.get_polarities()
        self.vector = np.zeros(len(polarities) + 1)
        if self.entity.get_class() == "Event":
            try:
                self.vector[polarities[self.entity.polarity]] = 1
            except KeyError:
                print(self.entity.polarity)
                self.vector[len(polarities)] = 1
            self.vector = scipy.sparse.csr_matrix(self.vector)


'''
One hot encoding of modality of the word (from input file)
'''


class ModalityFeatureVector(FeatureVector):
    def generate_vector(self):
        modalities = utils.get_modalities()
        self.vector = np.zeros(len(modalities) + 1)
        if self.entity.get_class() == "Event":
            try:
                self.vector[modalities[self.entity.modality]] = 1
            except KeyError:
                print(self.entity.modality)
                self.vector[len(modalities)] = 1
            self.vector = scipy.sparse.csr_matrix(self.vector)


'''
Do the two entities appear in the same paragraph?
'''


class SameParVector(RelationFeatureVector):
    def generate_vector(self):
        self.vector = [self.source.paragraph == self.target.paragraph]


'''
Do the two entities appear in the same sentence?
'''


class SameSentenceVector(RelationFeatureVector):
    def generate_vector(self):
        self.vector = [self.source.sentence == self.target.sentence]


'''
Specific feature vectors used in training and prediction
'''


class WordVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordFeatureVector(self.entity))
        self.features.append(POSFeatureVector(self.entity))
        self.features.append(CapitalFeatureVector(self.entity))
        self.features.append(DocTimeVector(self.entity))
        self.features.append(ModalityFeatureVector(self.entity))
        self.features.append(PolarityFeatureVector(self.entity))


class TimeRelationVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordVector(self.entity.source))
        self.features.append(WordVector(self.entity.target))
        self.features.append(SameParVector(self.entity.source, self.entity.target))
        self.features.append(SameSentenceVector(self.entity.source, self.entity.target))
