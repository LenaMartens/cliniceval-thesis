import nltk
from nltk.data import load
import numpy as np
import utils

dictionary = utils.get_dictionary()
lemma_dictionary = utils.get_lemma_dictionary()
'''
A vector denoting a single feature of a word
'''


class FeatureVector(object):
    def get_vector(self):
        self.generate_vector()
        return self.vector

    def generate_vector(self):
        self.vector = []

    def __init__(self, entity, document=None):
        self.entity = entity
        self.vector = []
        self.document = document


'''
A vector denoting a single feature of a relationship between two phrases
'''


class RelationFeatureVector(FeatureVector):
    def __init__(self, source, target):
        self.source = source
        self.vector = []
        self.target = target


'''
A vector consisting of several features. Is implemented as a list of FeatureVectors
'''


class ConcatenatedVector(object):
    def get_vector(self):
        self.generate_vector()
        return np.concatenate([x.get_vector() for x in self.features])

    def __init__(self, entity, document):
        self.entity = entity
        self.document = document
        self.features = list()

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
One hot encoding of the word in the dictionary of the lematized encountered words.
'''


class LemmaFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        word = utils.lemmatize_word(word)
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
        self.vector = np.asarray([self.entity.word.istitle()])


'''
What DOCTIME has the word been classified as?
'''


class DocTimeVector(FeatureVector):
    def generate_vector(self):
        doctimes = utils.get_doctimes()
        self.vector = np.zeros(len(doctimes) + 1)
        if self.entity.get_class() == "Event":
            try:
                self.vector[doctimes[self.entity.doc_time_rel]] = 1
            except KeyError or AttributeError:
                self.vector[len(doctimes)] = 1
        else:
            self.vector[len(doctimes)] = 1


tagdict = load('help/tagsets/upenn_tagset.pickle')
tag_list = list(tagdict.keys())

'''
One hot encoding of part of speech tag.
'''


class POSFeatureVector(FeatureVector):
    def generate_vector(self):
        sentence = self.document.get_sentence(self.entity)
        self.vector = np.zeros(len(tag_list))
        if sentence:
            # Pass whole sentence to get a better tagging
            tags = nltk.pos_tag(sentence)
            (word, tag) = tags[sentence.index(self.entity.word)]
            self.vector[tag_list.index(tag)] = 1


'''
One hot encoding of polarity of the word (from UMLS input file)
'''


class PolarityFeatureVector(FeatureVector):
    def generate_vector(self):
        polarities = utils.get_polarities()
        self.vector = np.zeros(len(polarities) + 1)
        if self.entity.get_class() == "Event":
            try:
                self.vector[polarities[self.entity.polarity]] = 1
            except KeyError or AttributeError:
                self.vector[len(polarities)] = 1


'''
One hot encoding of modality of the word (from UMLS input file)
'''


class ModalityFeatureVector(FeatureVector):
    def generate_vector(self):
        modalities = utils.get_modalities()
        self.vector = np.zeros(len(modalities) + 1)
        if self.entity.get_class() == "Event":
            try:
                self.vector[modalities[self.entity.modality]] = 1
            except KeyError:
                self.vector[len(modalities)] = 1
        else:
            self.vector[len(modalities)] = 1


'''
Do the two entities appear in the same paragraph?
'''


class SameParVector(RelationFeatureVector):
    def generate_vector(self):
        self.vector = np.asarray([self.source.paragraph == self.target.paragraph])


'''
Do the two entities appear in the same sentence?
'''


class SameSentenceVector(RelationFeatureVector):
    def generate_vector(self):
        self.vector = np.asarray([self.source.sentence == self.target.sentence])


'''
What is the distance between the two entities?
'''


# TODO: ONEHOT of normalisatie
class DistanceVector(RelationFeatureVector):
    def generate_vector(self):
        distance = abs(self.source.token - self.target.token)
        self.vector = np.asarray([distance / 30])


vector_length = 0


class EmptyVector(FeatureVector):
    def generate_vector(self):
        global vector_length
        self.vector = np.asarray(np.zeros(vector_length))


class BagOfWords(RelationFeatureVector):
    def __init__(self, document, source, target):
        RelationFeatureVector.__init__(self, source, target)
        self.document = document

    def generate_vector(self):
        span1 = self.source.span
        span2 = self.target.span
        words = self.document.get_words_inbetween(span1, span2)
        self.vector = np.zeros(len(dictionary) + 1)
        for word in words:
            try:
                self.vector[dictionary[word]] = 1
            except KeyError:
                self.vector[len(dictionary)] = 1


'''
Specific feature vectors used in training and prediction
'''


class WordVector(ConcatenatedVector):
    def generate_vector(self):
        global vector_length
        if self.entity is not None:
            self.features.append(WordFeatureVector(self.entity))
            self.features.append(LemmaFeatureVector(self.entity))
            self.features.append(CapitalFeatureVector(self.entity))
            self.features.append(POSFeatureVector(self.entity, document=self.document))
            self.features.append(DocTimeVector(self.entity))
            self.features.append(ModalityFeatureVector(self.entity))
            self.features.append(PolarityFeatureVector(self.entity))
            if vector_length == 0:
                for feature in self.features:
                    vector_length += len(feature.get_vector())
        else:
            self.features.append(EmptyVector(None))


class WordVectorWithContext(ConcatenatedVector):
    def generate_vector(self):
        left_neighbour = self.document.get_neighbour_entity(self.entity, -1)
        right_neighbour = self.document.get_neighbour_entity(self.entity, +1)
        self.features.append(WordVector(self.entity, self.document))
        self.features.append(WordVector(left_neighbour, self.document))
        self.features.append(WordVector(right_neighbour, self.document))


class TimeRelationVector(ConcatenatedVector):
    def generate_vector(self):
        self.features.append(WordVectorWithContext(self.entity.source, self.document))
        self.features.append(WordVectorWithContext(self.entity.target, self.document))
        self.features.append(SameParVector(self.entity.source, self.entity.target))
        self.features.append(SameSentenceVector(self.entity.source, self.entity.target))
        self.features.append(BagOfWords(self.document, self.entity.source, self.entity.target))


class ConfigurationFeature(ConcatenatedVector):
    def generate_vector(self):
        for entity in self.entity.get_top_entities("stack1", 3):
            self.features.append(WordVectorWithContext(self.document.entities[entity], document=self.document))
        for entity in self.entity.get_top_entities("stack2", 3):
            self.features.append(WordVectorWithContext(self.document.entities[entity], document=self.document))
        for entity in self.entity.get_top_entities("buffer", 3):
            self.features.append(WordVectorWithContext(self.document.entities[entity], document=self.document))
        stack = self.document.entities[self.entity.get_stack_head()]
        buffer = self.document.entities[self.entity.get_buffer_head()]
        self.features.append(SameParVector(stack, buffer))
        self.features.append(SameSentenceVector(stack, buffer))
        self.features.append(BagOfWords(self.document, stack, buffer))
