import bisect
import os
import xml.etree.ElementTree as ET
import utils
import re


class Document(object):
    def __init__(self, id):
        # array of strings
        self.sentences = ""

        # id to objects dictionary
        self.entities = {}
        self.relations = {}
        self.relation_mapping = {}
        self.id = id
	self.paragraph_delimiters = []
	
    def process_event(self, entity):
        id = entity.find("id").text
        id = id[:id.find('@')]

        span = [int(x) for x in re.split('[, ;]+', entity.find('span').text)]
        paragraph = bisect.bisect(self.paragraph_delimiters, span[0])
        word = self.get_word(span)
	if entity.find('type').text.lower() == "event":
            obj = Event(entity.find('properties'), span, word, id, paragraph)
            self.entities[id] = obj
        elif entity.find('type').text.lower().find("time") > -1:
            obj = Timex(entity.find('properties'), span, word, id, paragraph)
            self.entities[id] = obj
	
    def process_relation(self, relation):
        id = relation.find("id").text
        id = id[:id.find('@')]

        source_id = relation.find('properties').find('Source').text
        source = self.entities[source_id[:source_id.find('@')]]

        target_id = relation.find('properties').find('Target').text
        target = self.entities[target_id[:target_id.find('@')]]

        obj = Relation(source, relation.find('properties').find('Type').text, target, id=id)
        self.relations[id] = obj
        self.relation_mapping[source_id] = target_id

    def clear_relations(self):
        self.relations.clear()
        self.rel_id = 0

    def add_relation(self, source_id, sink_id):
        source = self.entities[source_id]
        sink = self.entities[sink_id]
        rel = Relation(source, "CONTAINS", sink, id=self.rel_id)
        self.relations[self.rel_id] = rel
        self.rel_id += 1

    def relation_exists(self, source, target):
        return source.id in self.relation_mapping and self.relation_mapping[source.id] == target.id

    def get_relations(self):
        return self.relations.values()

    def get_entities(self):
        return self.entities.values()

    def get_word(self, span):
        return self.sentences[span[0]:span[1]]

    def process_file(self, text_file):
        # Generate sentences
        f_handle = open(text_file, 'r')

        self.sentences = f_handle.read()
        self.paragraph_delimiters = [m.start() for m in re.finditer('\\n', self.sentences)]

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


class Event(object):
    @staticmethod
    def get_class():
        return "Event"

    def __init__(self, xml_dict, span, word, id, paragraph):
        self.paragraph = paragraph
        self.id = id
        self.span = span
        self.doc_time_rel = xml_dict.find('DocTimeRel').text
        self.type_class = xml_dict.find('Type').text
        self.degree = xml_dict.find('Degree').text
        self.polarity = xml_dict.find('Polarity').text
        self.contextual_modality = xml_dict.find('ContextualModality').text
        self.contextual_aspect = xml_dict.find('ContextualAspect').text
        self.permanence = xml_dict.find('Permanence').text
        self.word = word


class Timex(object):
    @staticmethod
    def get_class():
        return "TimeX3"

    def __init__(self, xml_dict, span, word, id, paragraph):
        self.paragraph = paragraph
        self.id = id
        self.span = span
        try:
            self.type_class = xml_dict.find('Class').text
        except AttributeError:
            self.type_class = "SECTIONTIME"
        self.word = word


class Relation(object):
    def __init__(self, source=None,
                 class_type="",
                 target=None,
                 positive=True,
                 id=0):
        self.id = id
        self.source = source
        self.class_type = class_type
        self.target = target
        self.positive = positive


def read_document(parent_directory, dir):
    # Give doc the correct ID
    doc = Document(dir)
    for file in os.listdir(os.path.join(parent_directory, dir)):
        file_path = os.path.join(parent_directory, dir, file)
        if file.find("Temporal") > -1:
            doc.process_annotations(file_path)
        elif file.find(".") == -1:
            doc.process_file(file_path)
    if(len(doc.get_entities()) == 0):
	print(doc.id)
    return doc


def read_all(directory):
    utils.load_dictionary()
    docs = []
    for dir in os.listdir(directory):
        doc = read_document(directory, dir)
        docs.append(doc)
        for k, entity in doc.entities.items():
            utils.add_word_to_dictionary(entity.word)
        utils.save_document(doc, dir)
    utils.save_dictionary()
    return docs


if __name__ == '__main__':
    from data import Document

    read_all(utils.dev)
