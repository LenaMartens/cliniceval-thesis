import logging
import dataset_attribute_experiment
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import oracle
import utils
from candidate_generation import generate_doctime_training_data, generate_constrained_candidates, doc_time_feature
from data import read_all
from feature import WordVectorWithContext, ConfigurationVector
import scipy.sparse
import random
from keras.models import Sequential
from keras.layers import Dense, Activation


class Classifier:
    def train(self, trainingdata):
        pass

    def predict(self, sample):
        pass

    def generate_training_data(self, docs):
        pass

    def __init__(self, trainingdata, class_to_fy=None):
        """
        :param trainingdata: docs
        :param class_to_fy:
        """
        self.class_to_fy = class_to_fy
        self.train(self.generate_training_data(trainingdata))


class LogisticRegression(Classifier):
    def generate_training_data(self, docs):
        return docs

    def train(self, generator):
        # PARTIAL FIT because of memory problems
        classes = None
        iterations=5
        for i in range(iterations):
            print("Iteration: " + str(i))
            for data in generator:
                X = [x.get_vector() for x in data]
                X = scipy.sparse.csr_matrix(X)
                Y = [getattr(x.entity, self.class_to_fy) for x in data]
                if classes == None:
                    classes = np.unique(Y)
                    print(classes)
                self.machine.partial_fit(X, Y, classes=classes)

    def predict(self, sample):
        # returns a log probability distribution
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict_proba(sample)

    def __init__(self, trainingdata, class_weights):
        # List of FeatureVectors
        self.class_to_fy = "positive"
        self.machine = linear_model.SGDClassifier(loss="log", penalty="l2")
        self.train(trainingdata)


class SupportVectorMachine(Classifier):
    def generate_training_data(self, docs):
        return generate_doctime_training_data(docs)

    def train(self, trainingdata):
        input = [x.get_vector() for x in trainingdata]
        output = [getattr(x.entity, self.class_to_fy) for x in trainingdata]
        input = scipy.sparse.csr_matrix(input)

        # BALANCED BECAUSE OF DATA BIAS + linear
        self.machine = svm.LinearSVC(class_weight='balanced')

        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict(sample)


class NNActions(Classifier):
    def generate_training_data(self, docs):
        for doc in docs:
            for paragraph in range(doc.get_amount_of_paragraphs()):
                entities = doc.get_entities(paragraph=paragraph)
                relations = doc.get_relations(paragraph=paragraph)
                for (configuration, action) in oracle.get_training_sequence(entities, relations):
                    feature = ConfigurationVector(configuration, doc).get_vector()
                    x = feature
                    y = utils.get_actions()[action]
                    yield (x, y)

    def train(self, trainingdata):
        """
        :param trainingdata: [X, Y], [[samples x features], [samples x 1] ], [feature vectors, indexes of actions]
        """
        X = trainingdata[0]
        print(X.shape)
        Y = trainingdata[1]
        Y = Y.reshape((-1, 1))
        print(Y.shape)
        model = Sequential()
        model.add(Dense(units=200, input_dim=X.shape[1]))
        model.add(Activation('softmax'))
        model.add(Dense(output_dim=4))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        model.fit(X, Y, epochs=5, batch_size=32)
        self.machine = model

    def predict(self, sample, doc):
        feature_vector = ConfigurationVector(sample, doc).get_vector()
        feature_vector = np.array(feature_vector)[np.newaxis]
        distribution = self.machine.predict(feature_vector)
        return distribution

def train_doctime_classifier(docs):
    svm = SupportVectorMachine(docs, "doc_time_rel")
    return svm


def feature_generator(docs, token_window, batch_size):
    logger = logging.getLogger('progress_logger')
    start = 0
    len_docs = len(docs)
    random.seed()
    random.shuffle(docs)
    while start < len_docs:
        logger.info("{start} out of {all}".format(start=start, all=len_docs))
        features = []
        end = min(start + batch_size, len(docs))
        for document in docs[start:end]:
            features.extend(generate_constrained_candidates(document, token_window))
        if features:
            logger.info("Features length:{l}".format(l=len(features)))
            yield features
        start += batch_size


def train_relation_classifier(docs, token_window):
    generator = feature_generator(docs, token_window, 5)
    candidate_counts = dataset_attribute_experiment.amount_of_candidates(docs, token_window)
    class_weights = {True:len(docs)/(2*candidate_counts[0]), False:len(docs)/(2*candidate_counts[1])}
    lr = LogisticRegression(generator, class_weights)
    return lr


def predict_DCT_document(document, model):
    document.clear_doc_time_rels()
    for entity in document.get_entities():
        if entity.get_class() == "Event":
            feature = doc_time_feature(entity, document)
            dct = model.predict(feature)
            entity.doc_time_rel = dct[0]
    return document


# Maybe this can be removed
def predict_DCT(documents, model=None):
    for document in documents:
        predict_DCT_document(document, model)
    return documents


if __name__ == '__main__':
    # because of pickle issues
    from classification import SupportVectorMachine, LogisticRegression

    docs = read_all(utils.dev)
    train_doctime_classifier(docs)
# features = generate_training_candidates(docs)
# lr = LogisticRegression(features)
# utils.save_model(lr, name="LogisticRegression_randomcandidate")
# for i in range(10):
#     print("predicted: " + str(lr.predict(features[i])) + " actual: " + str(features[i].entity.positive))
