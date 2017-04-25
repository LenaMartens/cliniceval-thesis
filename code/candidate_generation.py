from functools import partial

from data import Relation
from feature import WordVectorWithContext, TimeRelationVector, WordEmbeddingVectorWithContext


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


'''
MIGHT NEED IN FUTURE
def generate_training_candidates(documents, token_window):
    feature_vectors = []
    for document in documents:
        # Get positive candidates
        entities = list(document.get_entities())
        relations = document.get_relations()
        for relation in relations:
            feature_vectors.append(TimeRelationVector(relation, document))
        # Generate negative candidates (as many as there are positive)
        added = 0
        maxr = len(entities)
        relation_len = len(relations)
        added_dict = {}
        tried = 0
        while added < relation_len and tried < 100000:
            tried += 1
            [source_id, target_id] = random.sample(range(0, maxr), 2)
            if (source_id, target_id) not in added_dict:
                source = entities[source_id]
                target = entities[target_id]
                if not document.relation_exists(source, target) and abs(source.token - target.token) < token_window + 1:
                    relation = Relation(source=source, target=target, positive=False)
                    feature_vectors.append(TimeRelationVector(relation, document))
                    added += 1
                    added_dict[(source_id, target_id)] = True
    return feature_vectors
'''


def same_sentence(source, target):
    return source.sentence == target.sentence


def same_paragraph(source, target):
    return source.paragraph == target.paragraph


def in_window(window, source, target):
    return abs(source.token - target.token) < window + 1


def constrained_candidates(document, constraint):
    entities = list(document.get_entities())
    feature_vectors = []

    for entity1 in entities:
        for entity2 in entities:
            if entity1 is not entity2 and constraint(entity1, entity2):
                relation = Relation(source=entity1, target=entity2, positive=document.relation_exists(entity1, entity2))
                feature_vectors.append(TimeRelationVector(relation, document))

    return feature_vectors


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
