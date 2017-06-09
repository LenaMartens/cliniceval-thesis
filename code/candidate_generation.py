from functools import partial
import logging

from data import Relation
from feature import WordVectorWithContext, TimeRelationVector, WordEmbeddingVectorWithContext

"""
Candidate generation for the CR-task
"""

# DOCTIME candidates
def generate_doctime_training_data(documents):
    feature_vectors = []
    for document in documents:
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature_vectors.append(doc_time_feature(entity, document))
    return feature_vectors


def doc_time_feature(entity, document):
    return WordVectorWithContext(entity, document)



def same_sentence(source, target):
    return abs(source.sentence - target.sentence) < 2


def same_paragraph(source, target):
    return source.paragraph == target.paragraph


def in_window(window, source, target):
    return abs(source.token - target.token) < window + 1


def constrained_candidates(document, constraint):
    entities = list(document.get_entities())
    feature_vectors = [TimeRelationVector(x, document) for x in document.get_relations()]
    positive = len(feature_vectors)
    for entity1 in entities:
        for entity2 in entities:
            if entity1 is not entity2 and constraint(entity1, entity2) and not document.relation_exists(entity1, entity2):
                relation = Relation(source=entity1, target=entity2, positive=False)
                feature_vectors.append(TimeRelationVector(relation, document))
    return feature_vectors

# External method used by classification.py
def generate_constrained_candidates(document, token_window=30):
    return constrained_candidates(document, partial(in_window, token_window))


def generate_all_candidates(document):
    entities = list(document.get_entities())
    feature_vectors = []

    for entity1 in entities:
        for entity2 in entities:
            if entity1 is not entity2:
                relation = Relation(source=entity1, target=entity2, positive=False)
                feature_vectors.append(TimeRelationVector(relation, document))

    return feature_vectors
