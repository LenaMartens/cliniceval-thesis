import classification
import data
import utils
from annotator import GreedyAnnotator, InferenceAnnotator


class BaseProcedure(object):
    def __init__(self,
                 predict_path,
                 train_path="",
                 rel_classifier_path="",
                 doc_time_path="",
                 token_window=30,
                 greedy=False,
                 transitive=False):
        """
        :param predict_path: Path to corpus that needs to be annotated
        :param train_path: Path to training corpus (not required if models don't need to be retrained)
        :param token_window: Window in which candidates need to be generated
        :param rel_classifier: Path to Binary relation classification (YES/NO)
        :param doc_time_model: Path to Doctime classifier
        :param greedy: TRUE: use greedy decision making on binary classifications. FALSE: use ILP inference
        :param transitive: Close data transitively before training and
        """
        self.train_path = train_path
        self.predict_path = predict_path
        self.transitive = transitive
        self.token_window= token_window

        if greedy:
            self.annotator = GreedyAnnotator(token_window=token_window)
        else:
            self.annotator = InferenceAnnotator(token_window=token_window, transitive=transitive)

        if not rel_classifier_path:
            self.train_rel_classifier()
        else:
            self.annotator.model = utils.load_model(rel_classifier_path)

        if not doc_time_path:
            self.train_doctime()
        elif:
            self.doc_time_model = utils.load_model(doc_time_path)

    def train_doctime(self):
        if self.train_path:
            train_documents = data.read_all(self.train_path, transitive=self.transitive)
            self.doc_time_model = classification.train_doctime_classifier(train_documents)
        else:
            raise Exception("No path to training corpus provided")

    def train_rel_classifier(self):
        if self.train_path:
            train_documents = data.read_all(self.train_path, transitive=self.transitive)
            self.annotator.model = classification.train_relation_classifier(train_documents, self.token_window)
        else:
            raise Exception("No path to training corpus provided")

    def predict(self, filepath):
        documents = utils.test_document_generator(filepath)
        for doc in documents:
            self.doc_time_model.p
