import data
import oracle
import utils
from oracle import get_training_sequence

dev = data.read_all(utils.dev)
train = data.read_all(utils.train)


def treeless(documents):
    treefull = 0
    all = 0
    treeless = 0
    for document in documents:
        relations = document.relation_mapping
        parents = [x for (x, _) in relations.keys()]
        children = [x for (_, x) in relations.keys()]
        # how many relations in total are there?
        all += len(children)
        # how many children are not unique = how many children have multiple parents
        treefull += len(children) - len(set(children))
        # how many parents are not unique = how many parents have multiple children
        treeless += len(parents) - len(set(parents))
    print(treeless / all, treefull / all, all)


def samepar_relations(documents):
    amount_of_relations = 0
    amount_of_samepars = 0
    for document in documents:
        relations = document.get_relations()
        amount_of_relations += len(relations)
        for relation in relations:
            if abs(relation.source.paragraph - relation.target.paragraph) < 4:
                amount_of_samepars += 1
    print(amount_of_samepars, amount_of_relations)


def samesentence_relations(documents):
    amount_of_relations = 0
    amount_of_samesents = 0
    for document in documents:
        relations = document.get_relations()
        amount_of_relations += len(relations)
        for relation in relations:
            if relation.source.sentence == relation.target.sentence:
                amount_of_samesents += 1
    print(amount_of_samesents / amount_of_relations)


def within_amount_of_tokens(documents, amount=30):
    amount_of_relations = 0
    amount_of_withinamount = 0
    for document in documents:
        relations = document.get_relations()
        amount_of_relations += len(relations)
        for relation in relations:
            if abs(relation.source.token - relation.target.token) < amount + 1:
                amount_of_withinamount += 1
    print(amount_of_withinamount, amount_of_relations)


def amount_of_candidates(documents, amount=30):
    positive = 0
    negative = 0
    for document in documents:
        candidates = 0
        entities = document.get_entities()
        for i in entities:
            for j in entities:
                if i != j:
                    if abs(i.token - j.token) < amount + 1:
                        if document.relation_exists(i, j):
                            positive += 1
                        else:
                            negative += 1
    return positive, negative


def projective_trees(documents):
    all = 0
    relations_len = 0
    for document in documents:
        relations = document.get_relations()
        spans = []
        for relation in relations:
            source = relation.source
            target = relation.target
            begin = source.span[0]
            end = target.span[0]
            spans.append((begin, end))
        spans.sort(key=lambda x: x[0])
        non_projective = 0
        for i in range(len(spans) - 1):
            if spans[i + 1][0] <= spans[i][1] <= spans[i + 1][1]:
                non_projective += 1
        all+=non_projective
        relations_len+=len(relations)
        print(non_projective / (len(relations) + 1))
    print(all/relations_len)


def action_class_imbalance_paragraphs(documents):
    frequencies = {"left_arc": 0, "right_arc": 0, "no_arc": 0, "shift": 0}
    al = 0
    par=0
    for doc in documents:
        for paragraph in range(doc.get_amount_of_paragraphs()):
            par+=1
            entities = doc.get_entities(paragraph=paragraph)
            relations = doc.get_relations(paragraph=paragraph)
            for (configuration, action) in oracle.get_training_sequence(entities, relations, doc):
                frequencies[action] += 1
                al += 1
    print(frequencies, al, par)


def action_class_imbalance(documents):
    frequencies = {"left_arc": 0, "right_arc": 0, "no_arc": 0, "shift": 0}
    al = 0
    for doc in documents:
        entities = doc.get_entities()
        relations = doc.get_relations()
        for (configuration, action) in oracle.get_training_sequence(entities, relations, doc):
            frequencies[action] += 1
            al += 1
    print(frequencies, al, len(documents))


if __name__ == "__main__":
    #action_class_imbalance(dev)
    #action_class_imbalance(train)
    #action_class_imbalance_paragraphs(dev)
    # action_class_imbalance_paragraphs(train)
    # projective_trees(dev)
    print(amount_of_candidates(train, 25))
    print(amount_of_candidates(train, 10))
    #print(amount_of_candidates(dev))
    #treeless(dev)
    projective_trees(dev)
    
