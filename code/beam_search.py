import globals
import copy

import math

import utils


class Node(object):
    def __init__(self, parent, configuration, action, score):
        self.parent = parent
        self.action = action
        self.configuration = configuration
        self.score = score

def to_list(node):
    if node.parent is None:
        return []
    l = to_list(node.parent)
    l.append(node.configuration)
    return l


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
    gold_output = []

    while live_nodes and in_beam and l < k:
        l += 1
        in_beam = False
        try:
            (next_golden_config, next_golden_action) = next(golden_sequence)
        except StopIteration:
            break
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
        gold_output.append(next_golden_config)
        for node in live_nodes:
            if node.action == next_golden_action:
                in_beam = True
                break
    beam_sequences = []
    for node in dead_nodes + live_nodes:
        beam_sequences.append(to_list(node))
    return gold_output, beam_sequences


def score(previous, new):
    # smaller is better!!
    # negative log
    if new > 0:
        return previous.score - math.log(new)
    else:
        return previous.score - 1000
