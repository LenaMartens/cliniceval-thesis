import configparser
import logging
import os
import pickle
from sklearn.externals import joblib
from nltk.stem.wordnet import WordNetLemmatizer

import data

config = configparser.ConfigParser()
config.read('configuration.INI')
dev = config['DataLocation']['dev_corpus']
train = config['DataLocation']['train_corpus']
test = config['DataLocation']['test_corpus']
store_path = config['DataLocation']['doc_store']
model_path = config['ModelLocation']['models']

dictionary_path = config['DataLocation']['dictionary']
lemma_path = config['DataLocation']['lemma_dictionary']

outputpath = config['OutputLocation']['documents']
greedy_output_path = config['OutputLocation']['greedy']


def load_dictionary(dict_path):
    try:
        dict_file = open(dict_path, 'rb')
        dict = pickle.load(dict_file)
        dict_file.close()
    except Exception as e:
        logger = logging.getLogger('progress_logger')

        logger.error(e)
        dict = {}
    dictionary_id = len(dict)
    return dict, dictionary_id


(dictionary, dict_id) = load_dictionary(dictionary_path)
(lemma_dictionary, lemma_id) = load_dictionary(lemma_path)


def save_dictionary():
    with open(dictionary_path, 'wb') as dict_file:
        pickle.dump(dictionary, dict_file, protocol=2)

    with open(lemma_path, 'wb') as dict_file:
        pickle.dump(lemma_dictionary, dict_file, protocol=2)


def get_dictionary():
    return dictionary


def get_lemma_dictionary():
    return lemma_dictionary


def add_word_to_dictionary(word):
    global dictionary, dict_id, lemma_dictionary, lemma_id

    if word not in dictionary:
        dictionary[word] = dict_id
        dict_id += 1
    l_word = lemmatize_word(word)
    if l_word not in lemma_dictionary:
        lemma_dictionary[l_word] = lemma_id
        lemma_id += 1


def save_document(doc, id):
    with open(os.path.join(store_path, "doc_" + id), 'wb') as write_file:
        pickle.dump(doc, write_file, protocol=2)


def get_documents_from_file(filepath=store_path):
    documents = []
    for file in os.listdir(filepath):
        if file.find('doc') == -1:
            raise Exception(filepath + ' contains files that are not Documents')
        doc_file = open(os.path.join(filepath, file), 'rb')
        doc = pickle.load(doc_file)
        doc_file.close()
        documents.append(doc)
    return documents


def save_model(sv, name="anonymous"):
    logger = logging.getLogger('progress_logger')

    logger.info("Saving model: " + name)
    joblib.dump(sv, os.path.join(model_path, name))


def load_model(name):
    model = joblib.load(os.path.join(model_path, name))
    return model


def get_doctimes():
    return {"BEFORE": 0, "AFTER": 1, "OVERLAP": 2, "BEFORE/OVERLAP": 3}


def get_polarities():
    return {"NEG": 0, "POS": 1}


def get_modalities():
    return {"ACTUAL": 0, "HEDGED": 1, "HYPOTHETICAL": 2, "GENERIC": 3}


"""
GENERATORS FUELING THE ENGINE
"""


def document_generator(filepath=store_path):
    for file in os.listdir(filepath):
        if file.find('doc') != -1:
            with open(os.path.join(filepath, file), 'rb') as doc_file:
                yield pickle.load(doc_file)


def sentence_generator(filepath=dev):
    for direct in os.listdir(filepath):
        for file in os.listdir(os.path.join(filepath, direct)):
            if file.find(".") < 0:
                file_path = os.path.join(filepath, direct, file)
                with open(file_path) as f_handle:
                    for line in f_handle:
                        if not (line.startswith("[") and line.endswith("]\n")):
                            yield [x.lower() for x in line.split()]


# Returns Document objects with entities that can be annotated
def test_document_generator(filepath):
    for direct in os.listdir(filepath):
        yield data.read_document(parent_directory=filepath, dir=direct)


lmtzr = WordNetLemmatizer()


def lemmatize_word(word):
    return lmtzr.lemmatize(word)


def get_actions():
    """
    :return: Ground truth for what index is what action
    """
    return {"left_arc": 0, "right_arc": 1, "no_arc": 2, "shift": 3}


class Arc(object):
    def __init__(self, source, sink):
        """
        :param source: entity ID
        :param sink: entity ID
        """
        self.source = source
        self.sink = sink

    def __str__(self):
        return "{} -> {}".format(self.source, self.sink)
