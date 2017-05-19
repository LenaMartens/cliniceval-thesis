import logging
import objgraph
import os

import classification
import data
import global_norm_nn
import output
import utils
from annotator import GreedyAnnotator, InferenceAnnotator, TransitionAnnotator, BeamAnnotator
from oracle import NNOracle
from keras.models import load_model


class Procedure(object):
    def predict(self, filepath):
        logger = logging.getLogger('progress_logger')
        logger.info("Started prediction")
        documents = utils.test_document_generator(filepath)
        outputpath = self.generate_output_path(predict_path=filepath)
        for doc in documents:
            logger.info("Doc {id}".format(id=doc.id))
            if self.doc_time_model:
                doc = classification.predict_DCT_document(doc, self.doc_time_model)
            doc = self.annotator.annotate(doc)
            output.output_doc(doc, outputpath=outputpath)

    def evaluate(self, filepath):
        logger = logging.getLogger('progress_logger')
        logger.info("Evaluation:")
        outputpath = self.generate_output_path(filepath)
        anafora_command = "python -m anafora.evaluate -r {ref} -p {path} -x " \
                          "\"(?i).*clin.*Temp.*[.]xml$\"".format(ref=filepath, path=outputpath)
        os.system('cd ../anaforatools/;' + anafora_command + "> {path}".format(path=os.path.join(outputpath, "results.txt")))

    def generate_output_path(self, predict_path):
        return "shouldn't happen"


class BaseProcedure(Procedure):
    def __init__(self,
                 train_path="",
                 validation_path="",
                 retrain_rel=False,
                 retrain_dct=False,
                 rel_classifier_path="",
                 doc_time_path="",
                 token_window=30,
                 greedy=False,
                 transitive=False):
        """
        :param train_path: Path to training corpus (not required if models don't need to be retrained)
        :param token_window: Window in which candidates need to be generated
        :param rel_classifier_path: Path to Binary relation classification (YES/NO)
        :param doc_time_path: Path to Doctime classifier
        :param greedy: TRUE: use greedy decision making on binary classifications. FALSE: use ILP inference
        :param transitive: Close data transitively before training and
        """
        self.train_path = train_path
        self.transitive = transitive
        self.token_window = token_window
        self.greedy = greedy

        if greedy:
            self.annotator = GreedyAnnotator(token_window=token_window)
        else:
            self.annotator = InferenceAnnotator(token_window=token_window, transitive=transitive)

        if retrain_dct:
            self.doc_time_model = self.train_doctime(doc_time_path)
        else:
            self.doc_time_model = utils.load_model(doc_time_path)
        if retrain_rel:
            self.annotator.model = self.train_rel_classifier(rel_classifier_path)
        else:
            self.annotator.model = utils.load_model(rel_classifier_path)
        # evaluation
        if validation_path:
            docs = data.read_all(validation_path, transitive=transitive)
            dct = os.path.join(utils.model_path, doc_time_path + "_eval.txt")
            rel = os.path.join(utils.model_path, rel_classifier_path + "_eval.txt")
            with open(dct, 'w+') as file:
                eval_str = self.doc_time_model.evaluate(docs)
                print(eval_str)
                file.write(eval_str)
            with open(rel, 'w+') as file:
                eval_str = self.annotator.model.evaluate(docs)
                print(eval_str)
                file.write(eval_str)

    def train_doctime(self, save_path):
        logger = logging.getLogger('progress_logger')
        logger.info("Training doctime classifier")
        if self.train_path:
            logger.info("Reading documents")
            train_documents = data.read_all(self.train_path, transitive=self.transitive)
            logger.info("Started training")
            model = classification.train_doctime_classifier(train_documents)
            utils.save_model(model, name=save_path)
            return model
        else:
            raise Exception("No path to training corpus provided")

    def train_rel_classifier(self, save_path, validation=""):
        logger = logging.getLogger('progress_logger')
        logger.info("Training relation classifier")
        if self.train_path:
            logger.info("Reading documents")
            train_documents = data.read_all(self.train_path, transitive=self.transitive)
            logger.info("Started training")
            model = classification.train_relation_classifier(train_documents, self.token_window)
            utils.save_model(model, save_path)
            return model
        else:
            raise Exception("No path to training corpus provided")

    def generate_output_path(self, predict_path):
        p = os.path.split(predict_path)
        unique = "{decision}{window}{trans}{corpus}SENTENCEUMLS".format(decision=("Greedy" if self.greedy else "ILP"),
                                                            window=self.token_window,
                                                            trans=("Transitive" if self.transitive else ""),
                                                            corpus=p[-1])
        path = os.path.join(utils.outputpath, unique)
        return path


class TransitiveProcedure(Procedure):
    def __init__(self, train_path, global_norm=False, retrain = False, model_name="anon"):
        self.train_path = train_path
        self.global_norm = global_norm
        self.model_name = model_name
        self.retrain = retrain
        nn = self.train_network()
        oracle = NNOracle(nn)
        self.annotator = TransitionAnnotator(oracle=oracle)
        self.doc_time_model = None

    def generate_output_path(self, predict_path):
        unique = "model{model}".format(model=self.model_name)
        path = os.path.join(utils.outputpath, unique)
        return path

    def train_network(self):
        logger = logging.getLogger('progress_logger')
        logger.info("Training neural network")
        if self.train_path: 
            train_documents = None
            if self.retrain:
                logger.info("Reading documents")
                train_documents = data.read_all(self.train_path)
            logger.info("Started training")
            if self.global_norm:
                model = global_norm_nn.GlobalNormNN(train_documents, self.retrain, self.model_name)
            else:
                model = classification.NNActions(train_documents, self.retrain, self.model_name)
            return model
        else:
            raise Exception("No path to training corpus provided")
