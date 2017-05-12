import os
from keras.models import load_model
import logging
import random
import tensorflow as tf
import os
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
from keras.layers.merge import Add, Dot


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


def vectorize_config(x, doc):
    return np.asarray(ConfigurationVector(x, doc).get_vector())[np.newaxis]


actions = utils.get_actions()


def vectorize_action(action):
    em = np.zeros(len(actions))
    em[actions[action]] = 1
    return em


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
        The sequences i the beam also contain the golden sequence.
        A step consists of a feature vector of the state and a one-hot encoded vector of the decision taken
        """
        logger = logging.getLogger('progress_logger')

        while 1:
            random.seed()
            random.shuffle(docs)
            for doc in docs:
                for paragraph in range(doc.get_amount_of_paragraphs()):
                    entities = doc.get_entities(paragraph=paragraph)
                    if not entities:
                        break

                    relations = doc.get_relations(paragraph=paragraph)
                    golden_sequence = oracle.get_training_sequence(entities, relations, doc)
                    configuration = Configuration(entities, doc)
                    # Returns list of nodes
                    (golden_input, beam_inputs) = beam_search.in_beam_search(configuration, self,
                                                                             golden_sequence,
                                                                             beam=self.b, k=self.k)

                    empty_vector = vectorize_config(Configuration([], None), None)
                    empty_action = np.zeros(len(actions))
                    features = []
                    for node in golden_input:
                        features.append(vectorize_config(node.configuration, doc))
                        features.append(vectorize_action(node.action))
                    # Golden inputs padding
                    for i in range(self.k - len(golden_input)):
                        features.append(empty_vector)
                        features.append(empty_action)

                    # Add beam inputs with intermediate padding
                    for beam in beam_inputs:
                        for node in beam:
                            features.append(vectorize_config(node.configuration, doc))
                            features.append(vectorize_action(node.action))
                        # Golden inputs padding
                        for i in range(self.k - len(beam)):
                            features.append(empty_vector)
                            features.append(empty_action)

                    logger.info("Paragraph:" + str(paragraph) + ", sequence len=" + str(i))

                    # y_true is not used
                    yield (features, [empty_vector])

    def train(self, trainingdata):
        in_dim = len(ConfigurationVector(Configuration([], None), None).get_vector())

        # Shared model
        base_model = make_base_model(in_dim)
        self.base_model = base_model

        # All golden decisions = ONE SEQUENCE
        golden = []
        golden_inputs = []
        for i in range(self.k):
            name = "Golden" + str(i)
            # Configuration feature vector input
            state_input = Input(shape=(in_dim,), name=name)
            golden_inputs.append(state_input)
            # Decision input
            decision_input = Input(shape=(4,))
            golden_inputs.append(decision_input)
            # Probabilities of decisions under model
            distribution = base_model(state_input)
            # Decision activation
            output = Dot(0)(distribution, decision_input)
            golden.append(output)
        # Sum all probabilities in golden sequence
        golden_sum = Add()(golden)

        # All decisions in the beam = MULTIPLE SEQUENCES
        beam = []
        beam_inputs = []
        for i in range(self.b):
            sequence = []
            for j in range(self.k):
                name = "Beam{i}.{j}".format(i=i, j=j)
                # Configuration feature vector input
                state_input = Input(shape=(in_dim,), name=name)
                beam_inputs.append(state_input)
                # Decision input
                decision_input = Input(shape=(4,))
                beam_inputs.append(decision_input)
                # Probabilities of decisions under model
                distribution = base_model(state_input)
                # Decision activation
                output = Dot(0)(distribution, decision_input)
                sequence.append(output)
            beam.append(sequence)

        # Make the inner sums over all sequences in the beam
        inner_sequence_sums = list(map(Add(), beam))

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

        # Add beam inputs to all inputs
        for beams in beam_inputs:
            golden_inputs.extend(beams)

        model = Model(golden_inputs, output)

        self.machine = model
        self.graph = tf.get_default_graph()

        model.compile(loss=global_norm_loss, optimizer=SGD(lr=0.1))
        model.fit_generator(self.generate_training_data(trainingdata), verbose=1, epochs=2, steps_per_epoch=2, max_q_size=1)
        self.save()

    def predict(self, sample):
        with self.graph.as_default():
            feature_vector = ConfigurationVector(sample, sample.get_doc()).get_vector()
            feature_vector = np.array(feature_vector)[np.newaxis]
            distribution = self.base_model.predict(feature_vector)
        return distribution

    def save(self):
        self.base_model.save(os.path.join(utils.model_path, self.model_name))
    
    def load(self):
        self.base_model = load_model(os.path.join(utils.model_path, self.model_name))
    
    def __init__(self, trainingdata, pretrained=False, model_name="c00l_model"):
        """
        :param trainingdata: documents
        """
        self.machine = None
        self.base_model = None
        self.model_name = model_name
        if pretrained:
            self.load()
        else:
            self.train(trainingdata)




