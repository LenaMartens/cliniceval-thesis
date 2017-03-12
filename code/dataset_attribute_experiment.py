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


def within_amount_of_tokens(documents, amount=30):
    amount_of_relations = 0
    amount_of_withinamount = 0
    for document in documents:
        relations = document.get_relations()
        amount_of_relations += len(relations)
        for relation in relations:
            if abs(relation.source.token - relation.target.token) < amount + 1:
                amount_of_withinamount += 1
    print(amount_of_withinamount / amount_of_relations)


def amount_of_candidates(documents, amount=30):
    candidates = 0
    for document in documents:
        entities = document.get_entities()
        for i in entities:
            for j in entities:
                if i != j:
                    if abs(i.token - j.token) < amount + 1:
                        candidates += 1
    print(candidates)


if __name__ == "__main__":
    # treeless(train)
    # treeless(dev)
    # samepar_relations(train)
    # samepar_relations(dev)
    # samesentence_relations(train)
    # samesentence_relations(dev)
    for i in range(10, 15):
        print("amount: " + str(i))
        within_amount_of_tokens(train, i)
        amount_of_candidates(train, i)
        within_amount_of_tokens(dev, i)
        amount_of_candidates(dev, i)
        print("-------------------------------")
