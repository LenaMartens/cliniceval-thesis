import data
import utils

dev = data.read_all(utils.dev)
train = data.read_all(utils.train)


def treeless(documents):
    treefull = 0
    all = 0
    treeless = 0
    for document in documents:
        relations = document.relation_mapping
        children = list(relations.values())
        parents = list(relations.keys())
        # how many relations in total are there?
        all += len(children)
        # how many children are not unique = how many children have multiple parents
        treefull += len(children) - len(set(children))
        # how many parents are not unique = how many parents have multiple children
        treeless += len(parents) - len(set(parents))
    print(treeless, treefull, all)


def samepar_relations(documents):
    amount_of_relations = 0
    amount_of_samepars = 0
    for document in documents:
        relations = document.get_relations()
        amount_of_relations += len(relations)
        for relation in relations:
            if relation.source.paragraph == relation.target.paragraph:
                amount_of_samepars += 1
    print(amount_of_samepars / amount_of_relations)


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


if __name__ == "__main__":
    treeless(utils.train)
    treeless(utils.dev)
    samepar_relations(utils.train)
    samepar_relations(utils.dev)
    samesentence_relations(train)
    samesentence_relations(dev)
