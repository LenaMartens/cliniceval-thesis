from utils import Arc
from data import Entity

"""
A Covington style transition-based system.
"""


class Configuration:
    def __init__(self, entities, document):
        self.stack1 = [str(RootEntity())]
        self.stack2 = list()
        entities = list(entities)
        # Sort based on order in text.
        entities.sort(key=lambda x: x.span[0])
        self.buffer = [str(x) for x in entities]
        self.doc = document
        self.children_dict = {}
        self.arcs_dict = {}

    """
    CONDITIONS
    """
    def reachable(self, i, j, visited):
        if (i, j) in self.arcs_dict:
            return True
        visited[i] = True

        # all ks that are adjacent to i
        for k in [x[1] for x in self.arcs_dict.keys() if x[0] == i]:
            # don't visit nodes again
            if k not in visited:
                # recursive call :^)
                if self.reachable(k, j, visited):
                    return True
        return False

    def can_do_left_arc(self):
        if not self.stack1:
            return False
        i = self.stack1[-1]
        j = self.buffer[0]
        ROOT_CONDITION = (str(i) != "ROOT")
        HEAD_CONDITION = (i not in self.children_dict)
        NO_CYCLE_CONDITION = (j not in self.children_dict) or (not self.reachable(i, j, {}))
        return ROOT_CONDITION and HEAD_CONDITION and NO_CYCLE_CONDITION

    def can_do_right_arc(self):
        if not self.stack1:
            return False
        j = self.stack1[-1]
        i = self.buffer[0]
        ROOT_CONDITION = (str(i) != "ROOT")
        HEAD_CONDITION = (i not in self.children_dict)
        NO_CYCLE_CONDITION = (j not in self.children_dict) or (not self.reachable(i, j, {}))
        return ROOT_CONDITION and HEAD_CONDITION and NO_CYCLE_CONDITION

    def action_possible(self, action_str):
        if len(self.buffer) == 0:
            return False
        if action_str == "left_arc":
            return self.can_do_left_arc()
        if action_str == "right_arc":
            return self.can_do_right_arc()
        if action_str == "no_arc":
            return len(self.stack1) != 0
        return True

    """
    ACTIONS
    """

    # buffer to stack
    def left_arc(self):
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(j, i)] = True
        self.children_dict[i] = True
        self.stack2.append(i)

    # stack to buffer
    def right_arc(self):
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(i, j)] = True
        self.children_dict[j] = True
        self.stack2.append(i)

    def shift(self):
        b1 = self.buffer.pop(0)
        self.stack1.extend(self.stack2)
        self.stack1.append(b1)
        self.stack2 = []

    def no_arc(self):
        i = self.stack1.pop()
        self.stack2.append(i)

    """
    GETTERS
    """

    def get_buffer_head(self):
        return self.buffer[0]

    def get_stack_head(self):
        return self.stack1[-1]

    def get_arcs(self):
        arcs = []
        for (i, j) in self.arcs_dict.keys():
            if i != "ROOT":
                arcs.append(Arc(i, j))
        return arcs

    def get_top_entities(self, stack, amount):
        if stack == "stack1":
            if amount > len(self.stack1):
                return self.stack1 + [None] * (amount - len(self.stack1))
            else:
                return self.stack1[-amount:]
        if stack == "stack2":
            if amount > len(self.stack2):
                return self.stack2 + [None] * (amount - len(self.stack2))
            else:
                return self.stack2[-amount:]
        if stack == "buffer":
            if amount > len(self.buffer):
                return self.buffer + [None] * (amount - len(self.buffer))
            else:
                return self.buffer[-amount:]
        raise Exception(stack + " is not a valid option")

    def get_parent(self, id):
        if id is None:
            return None
        e = id.id
        if e in self.children_dict:
            for (p, k) in self.arcs_dict.keys():
                if e == k and p != "ROOT":
                    return p
        return None

    def get_doc(self):
        return self.doc

    """
    STATE CHECKERS
    """

    def on_stack(self, entity):
        return str(entity) in self.stack1

    def empty_buffer(self):
        return len(self.buffer) == 0

    def empty_stack(self):
        return len(self.stack1) == 0

    # String representation
    def __str__(self):
        buffer = [str(i) for i in self.buffer]
        stack1 = [str(i) for i in self.stack1]
        stack2 = [str(i) for i in self.stack2]
        return "buffer:{buffer}      stack1:{elements1}     stack2:{elements2}".format(buffer=buffer,
                                                                                       elements1=stack1,
                                                                                       elements2=stack2)

# The ROOT entity is a subclass of Entity and behaves the same.
class RootEntity(Entity):
    @staticmethod
    def get_class():
        return "ROOT"

    def __init__(self):
        pass

    def __str__(self):
        return "ROOT"
