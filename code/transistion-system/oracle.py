import utils
from data import read_all
from .covington_transistion import Configuration


class Oracle:
    def __init__(self, arcs):
        self.arcs = arcs

    def next_step(self, configuration):
        buffer = configuration.get_buffer_head()
        stack = configuration.get_stack_head()
        if (buffer, stack) in self.arcs:
            return "left_arc"
        if (stack, buffer) in self.arcs:
            return "right_arc"
        next = [x[1] for x in self.arcs.keys() if x[0] == buffer]
        next.extend([x[0] for x in self.arcs.keys() if x[1] == buffer])
        for n in next:
            if configuration.on_stack(n):
                return "no_arc"
        return "shift"


def get_training_sequence(entities, arcs):
    configuration = Configuration(entities)
    oracle = Oracle(arcs)

    sequence = []
    while not configuration.empty_buffer():
        function_string = oracle.next_step(configuration)
        sequence.append(function_string)
        getattr(configuration, function_string)()
    return sequence

if __name__ == '__main__':
    documents = read_all(utils.dev)
