import utils
import logging

from feature import WordEmbedding
from procedure import BaseProcedure, TransitiveProcedure

"""
BASE CONFIG
"""
retrain_DCT = False
retrain_REL = False
DCT_model_name = "SupportVectorMachine2.0Embeddings"
relation_model_name = "LogisticRegression2.0EmbeddingsSmallerTrainingWindow"
token_window = 30
transitive = False
greedy = False
"""
SHARED CONFIG
"""
train_path = utils.train
test_path = utils.dev


def complete_base():
    bp = BaseProcedure(train_path=train_path,
                       token_window=token_window,
                       retrain_rel=retrain_REL,
                       retrain_dct=retrain_DCT,
                       doc_time_path=DCT_model_name,
                       rel_classifier_path=relation_model_name,
                       greedy=greedy,
                       transitive=transitive)
    # Where the magic happens
    bp.predict(test_path)
    bp.evaluate(test_path)


def complete_transition():
    tp = TransitiveProcedure(train_path=train_path, global_norm=True)
    tp.predict(test_path)
    tp.evaluate(test_path)

if __name__ == "__main__":
    logger = logging.getLogger('progress_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('ilp.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Start')
    # train word embedding model
    WordEmbedding(None, True, utils.train, "../Models/WordEmbedding")
    complete_base()
