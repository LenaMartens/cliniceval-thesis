import data
import utils


def treeless(corpuspath):
    documents = data.read_all(corpuspath)
    treefull = 0
    all = 0
    treeless=0
    for document in documents:
        relations = document.relation_mapping
        children = list(relations.values())
        parents = list(relations.keys())
        all += len(children)
        treefull += len(children) - len(set(children))
        treeless += len(parents) - len(set(parents))
    print(treeless, treefull, all)


if __name__ == "__main__":
    treeless(utils.train)
    treeless(utils.dev)