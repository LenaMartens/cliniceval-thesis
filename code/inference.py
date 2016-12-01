import random

from gurobipy import *

import utils
from data import Relation, read_document
from feature import RelationFeatureVector

example_document = "C:\\Users\\lena\Documents\\THESIS\\THYME\\Test\\ID006_clinic_016"


def generate_prediction_candidates(document, amount=20):
    entities = list(document.get_entities())
    added = 0
    maxr = len(entities)
    added_dict = {}
    feature_vectors = []

    while added < amount:
        [source_id, target_id] = random.sample(range(0, maxr), 2)
        if (source_id, target_id) not in added_dict:
            source = entities[source_id]
            target = entities[target_id]
            relation = Relation(source=source, target=target, positive=False)
            feature_vectors.append(RelationFeatureVector(relation))
            added += 1
            added_dict[(source_id, target_id)] = True
    return feature_vectors


def generate_all_candidates(document):
    entities = list(document.get_entities())
    feature_vectors = []

    for entity1 in entities:
        for entity2 in entities:
            if entity1 is not entity2:
                relation = Relation(source=entity1, target=entity2, positive=False)
                feature_vectors.append(RelationFeatureVector(relation))

    return feature_vectors


def inference(document, logistic_model):
    candidates = generate_all_candidates(document)

    model = Model('Relations in document')
    model.Params.UpdateMode = 1
    # No output
    model.Params.OutputFlag = 0
    for candidate in candidates:
        probs = logistic_model.predict(candidate)
        # positive variable
        positive_var = model.addVar(vtype=GRB.BINARY, obj=probs[0][0],
                                    name="true: {}, {}".format(candidate.entity.source.id, candidate.entity.target.id))
        # negative variable
        negative_var = model.addVar(vtype=GRB.BINARY, obj=probs[0][1],
                                    name="false: {}, {}".format(candidate.entity.source.id, candidate.entity.target.id))
        model.addConstr(negative_var + positive_var == 1,
                        'only one label for {}, {}'.format(candidate.entity.source.id, candidate.entity.target.id))
    model.update()
    '''
    twee variablen per relatie -> niet-label en wel-label (makkelijker voor objectief) V
    CONSTRAINT -> maar 1 label per relatie V
    CONSTRAINT -> transitiviteit Cik - Cjk - Cij >= -1 V
    CONSTRAINT -> maar 1 positieve label per combinatie van entities -> of maar een richting bij de candidate generation
    '''
    entities = document.get_entities()
    for i in entities:
        for j in entities:
            for k in entities:
                if i is not j and j is not k and k is not i:
                    cik = model.getVarByName("true: {}, {}".format(i.id, k.id))
                    cjk = model.getVarByName("true: {}, {}".format(j.id, k.id))
                    cij = model.getVarByName("true: {}, {}".format(i.id, j.id))
                    model.addConstr(cik - cjk - cij >= -1, "transitivity")
    # maximize
    model.ModelSense = -1

    model.optimize()

    document.clear_relations()
    for var in model.getVars():
        if var.X == 1:
            str = var.VarName
            if str.startswith('true'):
                m = re.search(r'true: (.+?), (.+?)', str)
                if m:
                    document.add_relation(m.group(1), m.group(2))


def infer_relations_on_documents(documents, model=utils.load_model("LogisticRegression_randomcandidate")):
    for document in documents:
        print("Inference on {}".format(document.id))
        inference(document, model)


if __name__ == "__main__":
    logistic_model = utils.load_model("LogisticRegression_randomcandidate")
    for document in utils.get_documents_from_file():
        inference(document)
        utils.save_document(document, document.id)
