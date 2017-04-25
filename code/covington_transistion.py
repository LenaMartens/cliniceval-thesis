from utils import Arc
from data import Entity


class Configuration:
    def __init__(self, entities, document):
        # entity objects
        self.stack1 = [RootEntity()]
        self.stack2 = list()
        self.buffer = [str(x) for x in list(entities)]
        self.doc = document
        self.children_dict = {}
        self.arcs_dict = {}

    def reachable(self, i, j, visited):
        if (i, j) in self.arcs_dict:
            return True
        visited[i] = True

        for k in [x[1] for x in self.arcs_dict.keys() if x[0] == i]:
            if k in visited:
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
        NO_CYCLE_CONDITION = (j not in self.children_dict) or (not self.reachable(i, j, []))
        return ROOT_CONDITION and HEAD_CONDITION and NO_CYCLE_CONDITION

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

    def get_buffer_head(self):
        return self.buffer[0]

    def get_stack_head(self):
        return self.stack1[-1]

    def on_stack(self, entity):
        return str(entity) in self.stack1

    def empty_buffer(self):
        return len(self.buffer) == 0

    def empty_stack(self):
        return len(self.stack1) == 0

    def get_arcs(self):
        arcs = []
        for (i, j) in self.arcs_dict.keys():
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
        if id in self.children_dict:
            return list(self.arcs_dict.keys())[list(self.arcs_dict.values()).index(id)]
        else:
            return None

    def action_possible(self, action_str):
        if action_str == "left_arc":
            return self.can_do_left_arc()
        if action_str == "right_arc":
            return self.can_do_right_arc()
        return True

    def get_doc(self):
        return self.doc

    def __str__(self):
        buffer = [str(i) for i in self.buffer]
        stack1 = [str(i) for i in self.stack1]
        stack2 = [str(i) for i in self.stack2]
        return "buffer:{buffer}      stack1:{elements1}     stack2:{elements2}".format(buffer=buffer,
                                                                                       elements1=stack1,
                                                                                       elements2=stack2)


class RootEntity(Entity):
    @staticmethod
    def get_class():
        return "ROOT"

    def __init__(self):
        pass

    def __str__(self):
        return "ROOT"
