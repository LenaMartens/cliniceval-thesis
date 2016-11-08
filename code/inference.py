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
            if not document.relation_exists(source, target):
                relation = Relation(source=source, target=target, positive=False)
                feature_vectors.append(RelationFeatureVector(relation))
                added += 1
                added_dict[(source_id, target_id)] = True
    return feature_vectors


try:
    model = Model('Relations in document')
    model.Params.UpdateMode = 1

    document = read_document(example_document)
    candidates = generate_prediction_candidates(document)
    logistic_model = utils.load_model("LogisticRegression_randomcandidate")
    print(logistic_model.machine.get_params(True))
    variables = list()
    for candidate in candidates:
        probs = logistic_model.predict(candidate)
        variables.append(model.addVar(vtype=GRB.BINARY, obj=probs[0][0],
                                  name="{} -> {}".format(candidate.entity.source.id, candidate.entity.target.id)))
    model.update()

    model.addConstrs(quicksum(variables) == 10)
    model.ModelSense = -1
    model.optimize()
    for v in model.getVars():
        print('%s %g' % (v.varName, v.x))
    print(model.getObjective())
    print('Obj: %g' % model.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

