import copy

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
        if (str(buffer), str(stack)) in self.arcs:
            return "left_arc"
        if (str(stack), str(buffer)) in self.arcs:
            return "right_arc"

        next = [x[0] for x in self.arcs.keys() if x[1] == str(buffer)]

        # If entity on buffer has no parent, ROOT is parent
        if str(stack) == "ROOT" and not next:
            return "right_arc"

        next.extend([x[1] for x in self.arcs.keys() if x[0] == str(buffer)])
        for n in next:
            if configuration.on_stack(n):
                return "no_arc"
        return "shift"


class NNOracle(Oracle):
    def __init__(self, network):
        self.network = network

    def next_step(self, configuration):
        index = self.network.get_action(configuration)
        action = utils.get_actions()[index]
        return action


def get_training_sequence(entities, arcs):
    configuration = Configuration(entities)
    oracle = KnowingOracle(arcs)

    seqnce = []
    while not configuration.empty_buffer():
        function_string = oracle.next_step(configuration)
        seqnce.append(([], function_string))
        # applies function to configuration
        getattr(configuration, function_string)()
    return seqnce


if __name__ == '__main__':
    documents = read_all(utils.dev, transitive=False)
    for doc in documents:
        sequence = get_training_sequence(doc.get_entities(), doc.relation_mapping)

        print(len(doc.get_relations()), len([x for x in sequence if x[1] in ["left_arc", "right_arc"]]))
