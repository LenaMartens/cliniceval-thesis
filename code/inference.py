import random

from gurobipy import *

import classification
import output
import utils
from data import Relation, read_document
from feature import TimeRelationVector
from functools import partial
from guppy import hpy

hp = hpy()

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
                relation = Relation(source=entity1, target=entity2, positive=False)
                feature_vectors.append(TimeRelationVector(relation, document))

    return feature_vectors


def generate_all_candidates(document):
    entities = list(document.get_entities())
    feature_vectors = []

    for entity1 in entities:
        for entity2 in entities:
            if entity1 is not entity2:
                relation = Relation(source=entity1, target=entity2, positive=False)
                feature_vectors.append(TimeRelationVector(relation, document))

    return feature_vectors

def inference(document, logistic_model, token_window, transitive=False):
    candidates = constrained_candidates(document, partial(in_window, token_window))
    
    print(len(candidates))
    model = Model('Relations in document')
    # No output
    model.Params.OutputFlag = 0
    # Limit number of threads
    model.Params.Threads = 4
    model.Params.TimeLimit = 2
    '''
        CONSTRAINT -> maar 1 label per relatie V
        CONSTRAINT -> transitiviteit Cik - Cjk - Cij >= -1 V
        CONSTRAINT -> maar 1 positieve label per combinatie van entities -> of maar een richting bij de candidate generation
    '''
    for candidate in candidates:
        probs = logistic_model.predict(candidate)
        source = candidate.entity.source
        target = candidate.entity.target
        # positive variable
        positive_var = model.addVar(vtype=GRB.BINARY, obj=probs[0][1],
                                    name="true: {}, {}".format(source.id, target.id))
        # negative variable
        negative_var = model.addVar(vtype=GRB.BINARY, obj=probs[0][0],
                                    name="false: {}, {}".format(source.id, target.id))
        model.addConstr(negative_var + positive_var == 1,
                        'only one label for {}, {}'.format(source.id, target.id))
        if source.get_class() == "Event" and target.get_class() == "Event":
            if cannot_be_contained(source.doc_time_rel, target.doc_time_rel):
                model.addConstr(negative_var == 1,
                                '{} and {} do not have compatible doctimerels'.format(source.id,
                                                                                  target.id))
    
    model.update()

    entities = document.get_entities()
    for i in entities:
        parents_of_i = []
        for j in entities:
            cji = model.getVarByName("true: {}, {}".format(j.id, i.id))
            cij = model.getVarByName("true: {}, {}".format(i.id, j.id))

            if cji is not None and cij is not None:
                model.addConstr(cji + cij <= 1, "antisymmetry")
                if transitive:
                    for k in entities:
                        if i is not j and j is not k and k is not i:
                            cik = model.getVarByName("true: {}, {}".format(i.id, k.id))
                            cjk = model.getVarByName("true: {}, {}".format(j.id, k.id))
                            if cik is not None and cjk is not None and cij is not None:
                                model.addConstr(cik - cjk - cij >= -1, "transitivity")
            if cji is not None and not transitive:
                parents_of_i.append(cji)
        # Apply tree constraints if not transitive (only one parent)
        if parents_of_i:
            model.addConstr(sum(parents_of_i) <= 1, "treestructure")

    # maximize
    model.ModelSense = -1
    
    
    print(model.NumVars, model.NumConstrs)

    try:
        model.optimize()
    except GurobiError as e:
        print(e)
    original = len(document.get_relations())
    document.clear_relations()
    for var in model.getVars():
        if var.X == 1:
            str = var.VarName
            if str.startswith('true'):
                m = re.search(r'true: (.+?), (.+?)$', str)
                if m:
                    document.add_relation(m.group(1), m.group(2))
    print(original, len(document.get_relations()))


def greedy_decision(document, model, token_window, all=False):
    if all:
        candidates = generate_all_candidates(document)
    else:
        candidates = constrained_candidates(document, partial(in_window, token_window))

    document.clear_relations()
    for candidate in candidates:
        probs = model.predict(candidate)
        positive = probs[0][1]
        if positive > 0.7:
            document.add_relation(candidate.entity.source.id, candidate.entity.target.id)


def infer_relations_on_documents(documents, model, token_window):
    for i, document in enumerate(documents):
        print("Inference on {}".format(document.id) + ", number " + str(i))
        inference(document, model, token_window)
        print("Outputting document")
        output.output_doc(document)


def greedily_decide_relations(documents, model, token_window):
    for i, document in enumerate(documents):
        print("Greedy inference on {}".format(document.id) + ", number " + str(i))
        greedy_decision(document, model, token_window)
        print("Outputting document")
        output.output_doc(document, utils.greedy_output_path)


forbidden_love = {
    ("BEFORE", "AFTER"): True,
    ("AFTER", "BEFORE"): True,
    ("BEFORE", "OVERLAP"): True,
    ("AFTER", "OVERLAP"): True,
    ("BEFORE/OVERLAP", "AFTER"): True,
    ("BEFORE", "BEFORE/OVERLAP"): True,
    ("AFTER", "BEFORE/OVERLAP"): True,
}


def cannot_be_contained(container, containee):
    return (container, containee) in forbidden_love


if __name__ == "__main__":
    relation_model = classification.train_relation_classifier(utils.get_documents_from_file())
    for document in utils.get_documents_from_file():
        infer_relations_on_documents(document, relation_model)
