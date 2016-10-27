import os
import pickle
import xml.etree.ElementTree as ET
import configparser

import re

config = configparser.ConfigParser()
config.read('configuration.INI')
dev = config['DataLocation']['dev_corpus']
store_path = config['DataLocation']['doc_store']

dictionary_path = config['DataLocation']['dictionary']
dictionary = {}

dict_id = 0


def load_dictionary():
    global dictionary, dict_id

    try:
        dict_file = open(dictionary_path, 'rb')
        dictionary = pickle.load(dict_file)
    except:
        dictionary = {}
    print(len(dictionary))
    dict_id = len(dictionary)


def save_dictionary():
    print(len(dictionary))
    dict_file = open(dictionary_path, 'wb')
    pickle.dump(dictionary, dict_file)


def add_word_to_dictionary(word):
    global dictionary, dict_id

    if word not in dictionary:
        dictionary[word] = dict_id
        dict_id += 1


class Document:
    # doc ID
    id = 0

    # array of strings
    sentences = ""

    # id to objects dictionary
    entities = {}
    relations = {}

    def get_word(self, span):
        return self.sentences[span[0]:span[1]]

    def process_file(self, text_file):
        # Generate sentences
        f_handle = open(text_file, 'r')

        self.sentences = f_handle.read()

    def process_annotations(self, annotation_file):
        # Generate events and timex
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        entities_xml = root.find("annotations").findall("entity")
        for entity in entities_xml:
            self.process_event(entity)

        relations_xml = root.find("annotations").findall("relation")
        for relation in relations_xml:
            self.process_relation(relation)

    def __init__(self, id):
        self.id = id

    def process_event(self, entity):
        id = entity.find("id").text
        id = id[:id.find('@')]

        span = [int(x) for x in re.split('[, ;]+', entity.find('span').text)]
        word = self.get_word(span)
        if entity.find('type').text.lower() == "event":
            obj = Event(entity.find('properties'), span, word)
            self.entities[id] = obj
        elif entity.find('type').text.lower().startswith("time"):
            obj = Timex(entity.find('properties'), span, word)
            self.entities[id] = obj

    def process_relation(self, relation):
        id = relation.find("id").text
        id = id[:id.find('@')]

        source_id = relation.find('properties').find('Source').text
        if source_id.find('e') == -1:
            print(source_id)
            source = None
        else:
            source = self.entities[source_id[:source_id.find('@')]]

        target_id = relation.find('properties').find('Target').text
        target = self.entities[target_id[:target_id.find('@')]]

        obj = Relation(source, relation.find('properties').find('Type').text, target)
        self.relations[id] = obj


class Event:
    span = []
    doc_time_rel = ""
    type_class = ""
    degree = ""
    polarity = ""
    contextual_modality = ""
    contextual_aspect = ""
    permanence = ""
    word = ""

    def __init__(self, xml_dict, span, word):
        self.span = span
        self.doc_time_rel = xml_dict.find('DocTimeRel').text
        self.type_class = xml_dict.find('Type').text
        self.degree = xml_dict.find('Degree').text
        self.polarity = xml_dict.find('Polarity').text
        self.contextual_modality = xml_dict.find('ContextualModality').text
        self.contextual_aspect = xml_dict.find('ContextualAspect').text
        self.permanence = xml_dict.find('Permanence').text
        self.word = word


class Timex:
    span = []
    type_class = ""
    word = ""

    def __init__(self, xml_dict, span, word):
        self.span = span
        self.type_class = xml_dict.find('Class').text
        self.word = word


class Relation:
    source = None
    class_type = ""
    target = None

    def __init__(self, source=None,
                 class_type="",
                 target=None):
        self.source = source
        self.class_type = class_type
        self.target = target


if __name__ == '__main__':
    load_dictionary()
    for dir in os.listdir(dev):
        index = dir.rfind("_")
        id = dir[index + 1:index + 4]
        # Give doc the correct ID
        doc = Document(id)
        for file in os.listdir(os.path.join(dev, dir)):
            file_path = os.path.join(dev, dir, file)
            if file.find("Temporal-Relation") > -1:
                doc.process_annotations(file_path)
            elif file.find(".") == -1:
                doc.process_file(file_path)
        for k, entity in doc.entities.items():
            add_word_to_dictionary(entity.word)

        # Persist document in object structure
        write_file = open(os.path.join(store_path, "doc_" + id), 'wb')
        pickle.dump(doc, write_file)
    save_dictionary()
