import copy

import math

import utils


class Node(object):
    def __init__(self, parent, configuration, action, score):
        self.parent = parent
        self.action = action
        self.configuration = configuration
        self.score = score

    def to_list(self):
        if self.parent is None:
            return []
        return self.parent.to_list().append(self.configuration)


def beam_search(configuration, nn, beam=5):
    """
    Returns best sequence within beam.
    Beam = 1 generalizes to Greedy search.
    :param beam: size of beam
    :param nn: Neural network returning probability distribution
    :param configuration: Starting configuration
    :return: End node
    """

    dead_nodes = []
    live_nodes = [Node(None, configuration, None, 0)]
    actions = utils.get_actions()
    while live_nodes:
        new_nodes = []
        for node in live_nodes:
            distribution = nn.predict(node.configuration)
            for i, prob in enumerate(distribution[0]):
                action = list(actions.keys())[list(actions.values()).index(i)]
                if node.configuration.action_possible(action):
                    conf_copy = copy.deepcopy(node.configuration)
                    # applies action to config
                    getattr(conf_copy, action)()
                    if conf_copy.empty_buffer():
                        dead_nodes.append(Node(node, conf_copy, action, score(node, prob)))
                        beam -= 1
                    else:
                        new_nodes.append(Node(node, conf_copy, action, score(node, prob)))
        new_nodes.sort(key=lambda x: x.score)
        end = min(beam, len(new_nodes))
        live_nodes = new_nodes[:end]
    best = max(dead_nodes, key=lambda x: x.score)
    print(best.score)
    return best


def in_beam_search(configuration, nn, golden_sequence, k, beam=2):
    """
        Returns all beams the model predicts up until golden sequence
        falls outside of beam.
        Beam = 1 generalizes to Greedy search.
        :param k: max length of a sequence
        :param golden_sequence: training sequence
        :param beam: size of beam
        :param nn: Neural network returning probability distribution
        :param configuration: Starting configuration
        :return: All paths in beam
    """
    dead_nodes = []
    live_nodes = [Node(None, configuration, None, 0)]
    actions = utils.get_actions()
    in_beam = True
    l = 0
    golden_sequence = []

    while live_nodes and in_beam and l < k:
        l += 1
        new_nodes = []
        for node in live_nodes:
            distribution = nn.predict(node.configuration)
            for i, prob in enumerate(distribution[0]):
                action = list(actions.keys())[list(actions.values()).index(i)]
                if not node.configuration.empty_buffer() and node.configuration.action_possible(action):
                    conf_copy = copy.deepcopy(node.configuration)
                    # applies action to config
                    getattr(conf_copy, action)()
                    if conf_copy.empty_buffer():
                        dead_nodes.append(Node(node, conf_copy, action, score(node, prob)))
                        beam -= 1
                    else:
                        new_nodes.append(Node(node, conf_copy, action, score(node, prob)))
        new_nodes.sort(key=lambda x: x.score)
        end = min(beam, len(new_nodes))
        live_nodes = new_nodes[:end]
        in_beam = False
        (next_golden_config, next_golden_action) = golden_sequence.next()
        for node in live_nodes:
            if node.action == next_golden_action:
                in_beam = True
                golden_sequence.append(next_golden_config)
                break
    beam_sequences = []
    for node in dead_nodes + live_nodes:
        beam_sequences.extend(node.to_list())
    return golden_sequence, beam_sequences


def score(previous, new):
    # smaller is better!!
    # negative log
    return previous.score - math.log(new)
