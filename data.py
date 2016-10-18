import os
import xml.etree.ElementTree as ET
import configparser

config = configparser.ConfigParser()
config.read('configuration.INI')
dev = config['DataLocation']['dev_corpus']


class Document:
    # doc ID
    id = 0

    # array of strings
    sentences = []

    # id to objects dictionary
    entities = {}
    relations = {}

    def process_file(self, text_file):
        # Generate sentences
        f_handle = open(text_file, 'r')
        for line in f_handle:
            print(line)

    def process_annotations(self, annotation_file):
        # Generate events and timex
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        entities_xml = root.find("annotations").findall("entity")
        for entity in entities_xml:
            id = entity.find("id").text
            id = id[:id.find('@')]

            span = [int(x) for x in entity.find('span').text.split(',')]
            if entity.find('type').text.lower() == "event":
                obj = Event(entity.find('properties'), span)
                self.entities[id] = obj
            elif entity.find('type').text.lower().startswith("time"):
                obj = Timex(entity.find('properties'), span)
                self.entities[id] = obj

    def __init__(self, id):
        self.id = id


class Event:
    span = []
    doc_time_rel = ""
    type = ""
    degree = ""
    polarity = ""
    contextual_modality = ""
    contextual_aspect = ""
    permanence = ""

    def __init__(self, xml_dict, span):
        self.span = span
        self.doc_time_rel = xml_dict.find('DocTimeRel').text
        self.type = xml_dict.find('Type').text
        self.degree = xml_dict.find('Degree').text
        self.polarity = xml_dict.find('Polarity').text
        self.contextual_modality = xml_dict.find('ContextualModality').text
        self.contextual_aspect = xml_dict.find('ContextualAspect').text
        self.permanence = xml_dict.find('Permanence').text


class Timex:
    span = []
    type_class = ""

    def __init__(self, xml_dict, span):
        self.span = span
        self.type_class = xml_dict.find('Class').text


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
    for dir in os.listdir(dev):
        index = dir.rfind("_")
        doc = Document(dir[index + 1:index + 4])
        for file in os.listdir(os.path.join(dev, dir)):
            file_path = os.path.join(dev, dir, file)
            if file.find("Temporal-Relation") > -1:
                doc.process_annotations(file_path)
            elif file.find(".") == -1:
                doc.process_file(file_path)
