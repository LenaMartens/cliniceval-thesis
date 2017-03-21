class Configuration:
    def __init__(self, entities):
        # entity objects
        self.stack = list()
        self.buffer = entities
        self.dependency_arcs = list()

    def buffer_is_empty(self):
        return len(self.buffer) == 0

    def left_arc(self, label):
        assert (len(self.stack) > 1)
        s1 = self.stack.pop()
        s2 = self.stack.pop()
        self.dependency_arcs.append(Arc(s1, s2, label))
        self.stack.append(s1)

    def right_arc(self, label):
        assert (len(self.stack) > 1)
        s1 = self.stack.pop()
        s2 = self.stack.pop()
        self.dependency_arcs.append(Arc(s2, s1, label))
        # maybe this step is not needed -> do not remove s1
        self.stack.append(s2)

    def shift(self):
        b1 = self.buffer.pop(0)
        self.stack.append(b1)

    # remove from stack? maybe is needed -> only way to get rid of entity on stack is to make dependency?
    def remove(self):
        self.stack.pop()

    # expliciete actie?
    def stop(self):
        self.buffer = list()
    # Do we need to be able to have multiple arrows pointing to one thing?


class Arc:
    def __init__(self, source, sink, label):
        self.source = source
        self.sink = sink
        self.label = label

    def __str__(self):
        return "{} -> {}: {}".format(self.source, self.sink, self.label)
