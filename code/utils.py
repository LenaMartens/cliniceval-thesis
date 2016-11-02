import configparser
import os
import pickle

config = configparser.ConfigParser()
config.read('configuration.INI')
dev = config['DataLocation']['dev_corpus']
store_path = config['DataLocation']['doc_store']
model_path = config['ModelLocation']['models']

dictionary_path = config['DataLocation']['dictionary']
dictionary = {}

dict_id = 0


def load_dictionary():
    global dictionary, dict_id
    try:
        dict_file = open(dictionary_path, 'rb')
        dictionary = pickle.load(dict_file)
        dict_file.close()
    except Exception as e:
        print(e)
        dictionary = {}
    dict_id = len(dictionary)


load_dictionary()


def save_dictionary():
    with open(dictionary_path, 'wb') as dict_file:
        pickle.dump(dictionary, dict_file, protocol=2)


def get_dictionary():
    return dictionary


def add_word_to_dictionary(word):
    global dictionary, dict_id

    if word not in dictionary:
        dictionary[word] = dict_id
        dict_id += 1


def save_document(doc, id):
    with open(os.path.join(store_path, "doc_" + id), 'wb') as write_file:
        pickle.dump(doc, write_file, protocol=2)


def get_documents_from_file(filepath):
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
    with open(os.path.join(model_path, name+".pickle"), 'wb') as file:
        pickle.dump(sv, file, protocol=2)
