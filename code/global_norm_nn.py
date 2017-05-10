import logging
import random
import tensorflow as tf
from functools import reduce

import beam_search
import oracle
import utils
import numpy as np
from classification import Classifier
from covington_transistion import Configuration
from feature import ConfigurationVector
import keras.backend as K
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Merge
from keras.layers.merge import Add


def make_base_model(in_dim):
    model = Sequential()

    model.add(Dense(units=1024, input_dim=in_dim))
    model.add(Activation('softmax'))
    model.add(Dense(units=4))
    model.add(Activation('softmax'))

    return model


def global_norm_loss(y_true, y_pred):
    # y_true is a tuple:
    # (sum of predictions for golden sequence
    # ln of sum of sum of predictions of all beams)
    return y_pred

def negativeActivation(x):
    return -x

class GlobalNormNN(Classifier):
    """
    k: maximum length of sequence
    b: beam size
    """
    k = 50
    b = 3

    def generate_training_data(self, docs):
        """
        :yields: list of Configurations  that need to be evaluated
        List will be (b+1) * k long
        Say the golden sequence falls out of the beam after i steps:
        * 0:i items are first i golden steps
        * i:k are Empty vectors
        * b*k:(b*k)+i are first i steps of sequence for all sequences in beam
        * (b*k)+i:(b+1)*k are Empty vectors
        """
        logger = logging.getLogger('progress_logger')

        while 1:
            random.seed()
            random.shuffle(docs)
            model = self.machine
            for doc in docs:
                for paragraph in range(doc.get_amount_of_paragraphs()):
                    entities = doc.get_entities(paragraph=paragraph)
                    if not entities:
                        break

                    relations = doc.get_relations(paragraph=paragraph)
                    golden_sequence = oracle.get_training_sequence(entities, relations, doc)
                    configuration = Configuration(entities, doc)
                    # Returns list of configurations
                    (golden_input, beam_inputs) = beam_search.in_beam_search(configuration, self, 
                                                                             golden_sequence, 
                                                                             beam=self.b, k=self.k)

                    # How far it decoded before the golden sequence fell out of the beam
                    i = len(golden_input)
                    empty_vector = np.asarray(ConfigurationVector(Configuration([], None), None).get_vector()[np.newaxis])
                    features = [np.asarray(ConfigurationVector(x, doc).get_vector())[np.newaxis] for x in golden_input]
                    # Golden inputs padding
                    features.extend([empty_vector] * (self.k - i))
                    # Add beam inputs with intermediate padding
                    for beam in beam_inputs:
                        features.extend([np.asarray(ConfigurationVector(x, doc).get_vector())[np.newaxis] for x in beam])
                        features.extend([empty_vector] * (self.k - len(beam)))
                    logger.info("Paragraph:" + str(paragraph) + ", sequence len="+str(i))
                    # y_true is not used
                    yield (features, [empty_vector])

    def train(self, trainingdata):
        in_dim = len(ConfigurationVector(Configuration([], None), None).get_vector())

        # Shared model
        base_model = make_base_model(in_dim)
        self.base_model = base_model
        # Separate inputs
        # All golden decisions = ONE SEQUENCE
        golden_inputs = []
        for i in range(self.k):
            name = "Golden" + str(i)
            golden_inputs.append(Input(shape=(in_dim,), name=name))

        # Probabilities of decisions under model
        golden = list(map(base_model, golden_inputs))
        # Sum all probabilities in golden sequence
        golden_sum = Add()(golden)
        # All decisions in the beam = MULTIPLE SEQUENCES
        beam = []
        beam_inputs = []
        for i in range(self.b):
            sequence = []
            for j in range(self.k):
                name = "Beam{i}.{j}".format(i=i, j=j)
                sequence.append(Input(shape=(in_dim,), name=name))
            # Probabilities of decisions under model
            beam_inputs.append(sequence)
            sequence = list(map(base_model, sequence))
            beam.append(sequence)
        # Make the inner sums over all sequences in the beam
        inner_sequence_sums = []
        for sequence in beam:
            summation = Add()(sequence)
            inner_sequence_sums.append(summation)

        # Apply exp to all inner sums
        exp_activation = Activation(K.exp)
        inner_sequence_sums = list(map(exp_activation, inner_sequence_sums))
        # Sum all beam sequences together
        outer_sum = Add()(inner_sequence_sums)
        # Log activation
        outer_sum = Activation(K.log)(outer_sum)
        # Negate golden sum
        golden_sum = Activation(negativeActivation)(golden_sum)

        output = Add()([golden_sum, outer_sum])
        for beams in beam_inputs:
            golden_inputs.extend(beams)
        model = Model(golden_inputs, output)
        self.machine = model
        self.graph = tf.get_default_graph()
        model.compile(loss=global_norm_loss, optimizer=SGD(lr=0.1))
        model.fit_generator(self.generate_training_data(trainingdata), verbose=1, epochs=5, steps_per_epoch=1234, max_q_size=1)

    def predict(self, sample):
        with self.graph.as_default():
            feature_vector = ConfigurationVector(sample, sample.get_doc()).get_vector()
            feature_vector = np.array(feature_vector)[np.newaxis]
            distribution = self.base_model.predict(feature_vector)
        return distribution

    def save(self, filepath):
        self.machine.save(os.path.join(filepath, "C00l3st_model.h5"))
    
    def __init__(self, trainingdata):
        """
        :param trainingdata: documents
        """
        self.machine = None
        self.base_model = None
        self.train(trainingdata)
