class Configuration:
    def __init__(self, entities):
        # entity objects
        self.stack1 = list()
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

    def left_arc(self):
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(j, i)] = True
        self.children_dict[i] = True
        self.stack2.append(i)

    def right_arc(self):
        i = self.stack1.pop()
        j = self.buffer[0]
        self.arcs_dict[(j, i)] = True
        self.children_dict[j] = True
        self.stack2.append(i)

    def shift(self):
        b1 = self.buffer.pop(0)
        self.stack1.append(self.stack2)
        self.stack1.append(b1)

    def no_arc(self):
        i = self.stack1.pop()
        self.stack2.append(i)


class Arc:
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink

    def __str__(self):
        return "{} -> {}".format(self.source, self.sink)
