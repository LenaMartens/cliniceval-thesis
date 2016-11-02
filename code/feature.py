import numpy as np
import utils

dictionary = utils.get_dictionary()


class FeatureVector:
    entity = None
    vector = None

    def generate_vector(self):
        pass

    def __init__(self, entity):
        self.entity = entity
        self.generate_vector()


class WordFeatureVector(FeatureVector):
    def generate_vector(self):
        word = self.entity.word
        self.vector = np.zeros(len(dictionary))
	print(len(dictionary))
        self.vector[dictionary[word]] = 1

# if __name__ == "__main__":
#     f = WordFeatureVector(None)
