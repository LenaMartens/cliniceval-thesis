import copy
import _pickle as cPickle

import utils
from data import read_all
from covington_transistion import Configuration


class Oracle(object):
    def next_step(self, configuration):
        """

        :param configuration: Configuration
        :return: String describing the action that needs to be taken next
        """
        return ""


class KnowingOracle(Oracle):
    def __init__(self, arcs):
        self.arcs = arcs

    def next_step(self, configuration):
        if configuration.empty_stack():
            return "shift"

        buffer = configuration.get_buffer_head()
        stack = configuration.get_stack_head()
        for arc in self.arcs:
            if str(arc.source) == buffer and str(arc.target) == stack:
                return "left_arc"
            if str(arc.source) == stack and str(arc.target) == buffer:
                return "right_arc"

        next = [str(x.source) for x in self.arcs if str(x.target) == buffer]

        # If entity on buffer has no parent, ROOT is parent
        if str(stack) == "ROOT" and not next:
            return "right_arc"
        next.extend([str(x.target) for x in self.arcs if str(x.source) == buffer])
        for n in next:
            if configuration.on_stack(n):
                return "no_arc"
        return "shift"


class NNOracle(Oracle):
    # Regular old greedy parser
    def __init__(self, network):
        self.network = network

    def next_step(self, configuration):
        distribution = self.network.predict(configuration)
        actions = utils.get_actions()
        distribution = distribution.tolist()[0]
        en = list(enumerate(distribution))
        en.sort(key=lambda tup: tup[1])
        print(en)
        for (ind, val) in en[::-1]:
            action = list(actions.keys())[list(actions.values()).index(ind)]
            if configuration.action_possible(action):
                return action
        print("This should not print")
        return None


def get_training_sequence(entities, arcs, doc):
    configuration = Configuration(entities, doc)
    oracle = KnowingOracle(arcs)

    while not configuration.empty_buffer():
        function_string = oracle.next_step(configuration)
        conf_copy = cPickle.loads(cPickle.dumps(configuration, -1))
        yield (conf_copy, function_string)
        # applies function to configuration
        getattr(configuration, function_string)()


if __name__ == '__main__':
    documents = read_all(utils.dev, transitive=False)
    for doc in documents:
        sequence = get_training_sequence(doc.get_entities(), doc.get_relations(), doc)

        print(len(doc.get_relations()), len([x for x in sequence if x[1] in ["left_arc", "right_arc"]]))
