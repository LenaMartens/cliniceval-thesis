from utils import Arc
from data import Entity


class Configuration:
    def __init__(self, entities):
        # entity objects
        self.stack1 = [RootEntity()]
        self.stack2 = list()
        self.buffer = entities
        self.children_dict = {}
        self.arcs_dict = {}

    def reachable(self, i, j, visited):
        if (i, j) in self.arcs_dict:
            return True
        visited.append(i)

        for k in [x[1] for x in self.arcs_dict.keys if x[0] == i]:
            if visited.index(k) == -1:
                if self.reachable(k, j, visited):
                    return True
        return False

    def can_do_left_arc(self):
        i = self.stack1[-1]
        j = self.buffer[0]
        ROOT_CONDICTION = (i.get_class() != "ROOT")
        HEAD_CONDITION = (i not in self.children_dict)
        NO_CYCLE_CONDITION = (j not in self.children_dict) or (not self.reachable(i, j, []))
        return ROOT_CONDICTION and HEAD_CONDITION and NO_CYCLE_CONDITION

    def can_do_right_arc(self):
        j = self.stack1[-1]
        i = self.buffer[0]
        ROOT_CONDICTION = (i.get_class() != "ROOT")
        HEAD_CONDITION = (i not in self.children_dict)
        NO_CYCLE_CONDITION = (j not in self.children_dict) or (not self.reachable(i, j, []))
        return ROOT_CONDICTION and HEAD_CONDITION and NO_CYCLE_CONDITION

    # buffer to stack
    def left_arc(self):
        assert self.can_do_left_arc()
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(j, i)] = True
        self.children_dict[i] = True
        self.stack2.append(i)

    # stack to buffer
    def right_arc(self):
        assert self.can_do_right_arc()
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(j, i)] = True
        self.children_dict[j] = True
        self.stack2.append(i)

    def shift(self):
        b1 = self.buffer.pop(0)
        self.stack1.extend(self.stack2)
        self.stack1.append(b1)

    def no_arc(self):
        i = self.stack1.pop()
        self.stack2.append(i)

    def get_buffer_head(self):
        return self.buffer[0]

    def get_stack_head(self):
        return self.stack1[-1]

    def on_stack(self, entity):
        return self.stack1.index(entity) > -1

    def empty_buffer(self):
        return len(self.buffer) == 0

    def empty_stack(self):
        return len(self.buffer) == 0

    def get_arcs(self):
        arcs = []
        for (i, j) in self.arcs_dict.keys():
            arcs.append(Arc(i, j))
        return arcs


class RootEntity(Entity):
    @staticmethod
    def get_class():
        return "ROOT"
