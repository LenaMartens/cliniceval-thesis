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
        self.relations = []
        self.relation_mapping = {}
        self.id = id
        self.rel_id = 0
        self.paragraph_delimiters = []
        self.sentence_delimiters = []
        self.token_delimiters = []
        self.sorted_entity_ids = []

    def clear_relations(self):
        self.relations.clear()

    '''
    Relation adding helper methods
    '''

    def add_relation(self, source_id, sink_id):
        source = self.entities[source_id]
        sink = self.entities[sink_id]
        rel = Relation(source, "CONTAINS", sink, id=len(self.relations))
        self.relations.append(rel)
        self.relation_mapping[(source_id, sink_id)] = True

    def relation_exists(self, source, target):
        return (source.id, target.id) in self.relation_mapping

    '''
    Getters
    '''

    def get_relations(self):
        return self.relations

    def get_entities(self):
        return self.entities.values()

    def get_word(self, span):
        return self.sentences[span[0]:span[1]]

    def get_neighbour_entity(self, entity, direction):
        '''
        :param entity: Entity to get neighbour of
        :param direction: -1 for left, +1 for right
        :return: neighbouring entity based on span
        '''
        place = self.sorted_entity_ids.index(entity.id)
        try:
            return self.sorted_entity_ids[place + direction]
        except IndexError:
            return None

    '''
    Processing methods describing how to turn the raw input to objects
    '''

    def process_event(self, entity):
        id = entity.find("id").text
        id = id[:id.find('@')]

        span = [int(x) for x in re.split('[, ;]+', entity.find('span').text)]
        paragraph = bisect.bisect(self.paragraph_delimiters, span[0]) - 1
        sentence = bisect.bisect(self.sentence_delimiters, span[0]) - 1
        token = bisect.bisect(self.token_delimiters, span[0]) - 1
        word = self.get_word(span)
        obj = None
        if entity.find('type').text.lower() == "event":
            obj = Event(entity.find('properties'), span, word, id, paragraph, sentence, token)
        elif entity.find('type').text.lower().find("time") > -1:
            obj = Timex(entity.find('properties'), span, word, id, paragraph, sentence, token)
        self.entities[id] = obj
        self.sorted_entity_ids.append(obj.id)

    def process_relation(self, relation):
        type = relation.find('properties').find('Type').text
        if type.lower() == "contains":
            id = relation.find("id").text
            id = id[:id.find('@')]

            source_id = relation.find('properties').find('Source').text
            source_id = source_id[:source_id.find('@')]

            target_id = relation.find('properties').find('Target').text
            target_id = target_id[:target_id.find('@')]

            self.add_relation(source_id, target_id)

    def process_file(self, text_file):
        # Generate sentences
        f_handle = open(text_file, 'r')

        self.sentences = f_handle.read()
        self.paragraph_delimiters = [m.start() for m in re.finditer('\\n', self.sentences)]
        self.sentence_delimiters = [m.start() for m in re.finditer('\.', self.sentences)]
        self.token_delimiters = [m.start() for m in re.finditer('\s', self.sentences)]

        f_handle.close()

    def process_annotations(self, annotation_file):
        # Generate events and timex
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        entities_xml = root.find("annotations").findall("entity")
        for entity in entities_xml:
            self.process_event(entity)
        # After all entities have been added, sort id list according to span
        self.sorted_entity_ids.sort(key=lambda x: self.entities[x].span[0])
        relations_xml = root.find("annotations").findall("relation")
        for relation in relations_xml:
            self.process_relation(relation)

    '''
    After reading in the document, add transitive relations if not already there
    '''

    def close_transitivity(self):
        entities = self.get_entities()
        mapping = self.relation_mapping
        for i in entities:
            for j in entities:
                for k in entities:
                    if i is not j and j is not k and k is not i:
                        if (i.id, j.id) in mapping and (j.id, k.id) in mapping and (i.id, k.id) not in mapping:
                            self.add_relation(i.id, k.id)


'''
Cmp function for sorting of entities
'''


def entity_span_cmp(self, id1, id2):
    entity1 = self.entities[id1]
    entity2 = self.entities[id2]
    return entity1.span[0] - entity2.span[0]


'''
Data classes
'''


class Event(object):
    @staticmethod
    def get_class():
        return "Event"

    def __init__(self, xml_dict, span, word, id, paragraph, sentence, token):
        self.paragraph = paragraph
        self.sentence = sentence
        self.token = token
        self.id = id
        self.span = span
        self.doc_time_rel = xml_dict.find('DocTimeRel').text
        self.type_class = xml_dict.find('Type').text
        self.degree = xml_dict.find('Degree').text
        self.polarity = xml_dict.find('Polarity').text
        self.modality = xml_dict.find('ContextualModality').text
        self.contextual_aspect = xml_dict.find('ContextualAspect').text
        self.permanence = xml_dict.find('Permanence').text
        self.word = word


class Timex(object):
    @staticmethod
    def get_class():
        return "TimeX3"

    def __init__(self, xml_dict, span, word, id, paragraph, sentence, token):
        self.paragraph = paragraph
        self.sentence = sentence
        self.id = id
        self.span = span
        self.token = token
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


'''
Helper reading methods for external use
'''


def read_document(parent_directory, dir):
    # Give doc the correct ID
    doc = Document(dir)
    for file in os.listdir(os.path.join(parent_directory, dir)):
        file_path = os.path.join(parent_directory, dir, file)
        if file.find("Temporal") > -1:
            doc.process_annotations(file_path)
        elif file.find(".") == -1:
            doc.process_file(file_path)
    doc.close_transitivity()
    return doc


def read_all(directory):
    utils.load_dictionary(utils.lemma_path)
    utils.load_dictionary(utils.dictionary_path)
    docs = []
    for dir in os.listdir(directory):
        doc = read_document(directory, dir)
        docs.append(doc)
        for k, entity in doc.entities.items():
            utils.add_word_to_dictionary(entity.word)
    utils.save_dictionary()
    return docs


if __name__ == '__main__':
    from data import Document

    read_all(utils.train)
    read_all(utils.dev)
