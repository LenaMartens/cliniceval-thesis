import functools

from tqdm import *
from itertools import tee
import logging

import beam_search
import dataset_attribute_experiment
from sklearn import svm, linear_model
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import oracle
import utils
from candidate_generation import generate_doctime_training_data, generate_constrained_candidates, doc_time_feature
from covington_transistion import Configuration
from data import read_all
from feature import WordVectorWithContext, ConfigurationVector, TimeRelationVector
import scipy.sparse
import random
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics


class Classifier:
    def train(self, trainingdata):
        pass

    def predict(self, sample):
        pass

    def generate_training_data(self, docs):
        pass

    def evaluate(self, validation):
        pass

    def __init__(self, trainingdata, class_to_fy=None):
        """
        :param trainingdata: docs
        :param class_to_fy:
        """
        self.class_to_fy = class_to_fy
        self.train(self.generate_training_data(trainingdata))


class LogisticRegression(Classifier):
    def evaluate(self, validation):
        y_true = []
        y_pred = []
        for doc in validation:
            for candidate in generate_constrained_candidates(doc, 30):
                y_true.append(candidate.entity.positive)
                distribution = self.predict(candidate)
                if distribution[0][0] > distribution[0][1]:
                    y_pred.append(False)
                else:
                    y_pred.append(True)

        return (classification_report(y_true, y_pred)) + "\n" + \
               'F1 score:{}'.format(f1_score(y_true, y_pred)) + "\n" + \
               'confusion matrix:{}'.format(confusion_matrix(y_true, y_pred))

    def generate_training_data(self, docs):
        return docs

    def train(self, generator):
        # PARTIAL FIT because of memory problems
        if self.batches:
            classes = None
            iterations = 2
            for i in range(iterations):
                print("Iteration: " + str(i))
                for data in generator:
                    X = [x.get_vector() for x in data]
                    X = scipy.sparse.csr_matrix(X)
                    Y = [getattr(x.entity, self.class_to_fy) for x in data]
                    if not classes:
                        classes = np.unique(Y)
                    self.machine.partial_fit(X, Y, classes=classes)
        else:
            data = [item for sublist in generator for item in sublist]
            input = [x.get_vector() for x in data]
            output = [getattr(x.entity, self.class_to_fy) for x in data]
            input = scipy.sparse.csr_matrix(input)
            self.machine.fit(input, output)

    def predict(self, sample):
        # returns a log probability distribution
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict_proba(sample)

    def __init__(self, trainingdata, token_window, batches=False):
        # List of FeatureVectors
        self.class_to_fy = "positive"
        self.token_window = token_window
        self.batches = batches
        if batches:
            self.machine = linear_model.SGDClassifier(loss="log", penalty="l2")
        else:
            self.machine = linear_model.LogisticRegression()
        self.train(trainingdata)


class SupportVectorMachine(Classifier):
    def evaluate(self, validation):
        y_true = []
        y_pred = []
        for doc in tqdm(validation):
            for entity in doc.get_entities():
                if entity.get_class() == "Event":
                    y_true.append(entity.doc_time_rel)
                    entity.doc_time_rel = None
                    y_pred.append(self.predict(WordVectorWithContext(entity, doc))[0])
        return (classification_report(y_true, y_pred)) + "\n" + \
               'F1 score:{}'.format(f1_score(y_true, y_pred, average='weighted')) + "\n" + \
               'confusion matrix:\n{}'.format(confusion_matrix(y_true, y_pred))

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


def global_norm_loss(y_true, y_pred):
    # y_true is a tuple:
    # (sum of predictions for golden beam
    # ln of sum of sum of predictions of all beams)
    return - y_true[0] + K.logsumexp(y_true[1])


def early_update_generator(docs, model):
    while 1:
        random.seed()
        random.shuffle(docs)
        for doc in docs:
            for paragraph in range(doc.get_amount_of_paragraphs()):
                entities = doc.get_entities(paragraph=paragraph)
                relations = doc.get_relations(paragraph=paragraph)
                golden_sum = 0
                golden_sequence = []

                for (configuration, action) in oracle.get_training_sequence(entities, relations, doc):
                    golden_sequence.append(action)
                    index = utils.get_actions()[action]
                    distribution = model.predict(configuration)
                    value = distribution[index]
                    golden_sum += value

                configuration = Configuration(entities, doc)
                beam_list = beam_search.in_beam_search(configuration, model, golden_sequence)
                list_of_beam_values = []
                for beam in beam_list:
                    list_of_beam_values.append(beam.score)
                yield ([ConfigurationVector(configuration, doc)],
                       [(golden_sum, list_of_beam_values)])


class NNActions(Classifier):
    def generate_training_data(self, docs, batch_size=100):
        while 1:
            x_train = []
            y_train = []
            random.seed()
            random.shuffle(docs)
            for doc in docs:
                for paragraph in range(doc.get_amount_of_paragraphs()):
                    entities = doc.get_entities(paragraph=paragraph)
                    relations = doc.get_relations(paragraph=paragraph)
                    for (configuration, action) in oracle.get_training_sequence(entities, relations, doc):
                        feature = ConfigurationVector(configuration, doc).get_vector()
                        x_train.append(feature)
                        y_train.append(utils.get_actions()[action])
                        if len(x_train) == batch_size:
                            yield (np.vstack(x_train), np.vstack(y_train))
                            x_train = []
                            y_train = []

    def train(self, trainingdata):
        """
        :param trainingdata: batch generator
        """
        model = Sequential()
        trainingdata = trainingdata(model)
        t, t_backup = tee(trainingdata)
        (x, y) = next(t)
        in_dim = len(x[0])

        model.add(Dense(units=200, input_dim=in_dim))
        model.add(Activation('softmax'))
        model.add(Dense(units=4))
        model.add(Activation('softmax'))
        model.compile(loss=global_norm_loss,
                      optimizer='sgd',
                      metrics=[metrics.categorical_accuracy])

        model.fit_generator(t_backup, verbose=1, epochs=2, steps_per_epoch=2)
        self.machine = model

    def predict(self, sample):
        feature_vector = ConfigurationVector(sample, sample.get_doc()).get_vector()
        feature_vector = np.array(feature_vector)[np.newaxis]
        distribution = self.machine.predict(feature_vector)
        return distribution

    def __init__(self, training_data, global_norm=False):
        self.machine = None
        if global_norm:
            self.train(functools.partial(early_update_generator,training_data)
        else:
            self.train(self.generate_training_data(training_data))


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
            features.extend(generate_constrained_candidates(document, 10))
        if features:
            logger.info("Features length:{l}".format(l=len(features)))
            yield features
        start += batch_size


def train_relation_classifier(docs, token_window):
    generator = feature_generator(docs, token_window, 10)
    lr = LogisticRegression(generator, token_window, False)
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
