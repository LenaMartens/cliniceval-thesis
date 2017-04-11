import utils
import logging

from procedure import BaseProcedure, TransitiveProcedure

"""
BASE CONFIG
"""
retrain_DCT = False
retrain_REL = True
DCT_model_name = "SupportVectorMachine1.0"
relation_model_name = "LogisticRegression1.0"
token_window = 30
transitive = True
greedy = False
"""
SHARED CONFIG
"""
train_path = utils.train
test_path = utils.dev


def complete_base():
    DCT = DCT_model_name
    relation_model = relation_model_name
    if retrain_DCT:
        DCT = ""
    if retrain_REL:
        relation_model = ""
    bp = BaseProcedure(train_path=train_path,
                       token_window=token_window,
                       doc_time_path=DCT,
                       rel_classifier_path=relation_model,
                       greedy=greedy,
                       transitive=transitive)
    if retrain_DCT:
        utils.save_model(bp.doc_time_model, name=DCT_model_name)
    if retrain_REL:
        utils.save_model(bp.annotator.model, name=relation_model_name)
    # Where the magic happens
    bp.predict(test_path)
    bp.evaluate(test_path)


def complete_transition():
    tp = TransitiveProcedure(train_path=train_path)
    tp.predict(test_path)
    tp.evaluate(test_path)

if __name__ == "__main__":
    logger = logging.getLogger('progress_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('progress.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Start')
    complete_base()
