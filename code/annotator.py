from inference import inference, greedy_decision

from nns.covington_transistion import Configuration


def add_arcs_to_document(arcs, document):
    for arc in arcs:
        document.add_relation(arc.source, arc.sink)
    return document


class RelationAnnotator(object):
    def __init__(self, token_window, model=None):
        """
        :param model: model that predict probability of a relation existing
        :param token_window: window for candidate generation
        """
        self.model = model
        self.token_window = token_window

    def get_arcs(self, document):
        return []

    def annotate(self, document):
        document.clear_relations()
        arcs = self.get_arcs(document)
        return add_arcs_to_document(arcs, document)


class InferenceAnnotator(RelationAnnotator):
    def __init__(self, token_window, transitive, model=None):
        """
        :param model: model that predict probability of a relation existing
        :param token_window: window for candidate generation
        :param transitive: boolean indicating whether or not to apply transitivity constraints
        """
        super().__init__(model=model, token_window=token_window)
        self.transitive = transitive

    def get_arcs(self, document):
        return inference(document, self.model, self.token_window, self.transitive)


class GreedyAnnotator(RelationAnnotator):
    def get_arcs(self, document):
        return greedy_decision(document, self.model, self.token_window)


class TransitionAnnotator(RelationAnnotator):
    def __init__(self, oracle):
        """
        :param oracle: Oracle that decides sequence of steps
        """
        self.oracle = oracle

    def get_arcs(self, document):
        arcs = []
        for paragraph in range(document.get_paragraph_amount()):
            entities = document.get_entities(paragraph=paragraph)
            if entities:
                configuration = Configuration(entities)
                while not configuration.empty_buffer():
                    action_string = self.oracle.next_step(configuration)
                    # applies function to configuration
                    getattr(configuration, action_string)()
                arcs.extend(configuration.get_arcs())
        return arcs


class Arc:
    def __init__(self, source, sink):
        """
        :param source: entity ID
        :param sink: entity ID
        """
        self.source = source
        self.sink = sink

    def __str__(self):
        return "{} -> {}".format(self.source, self.sink)
