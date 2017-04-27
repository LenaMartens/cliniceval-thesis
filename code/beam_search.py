import numpy as np
import copy
import utils


class Node(object):
    def __init__(self, parent, configuration, action, score):
        self.parent = parent
        self.action = action
        self.configuration = configuration
        self.score = score


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


def score(previous, new):
    # smaller is better!!
    # negative log
    return previous.score - np.log(new)
