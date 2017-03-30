import utils

from procedure import BaseProcedure
"""
BASE CONFIG
"""
retrain_DCT = False
retrain_REL = True
DCT_model_name = "SupportVectorMachine1.0"
relation_model_name = "LogisticRegression1.0"
if retrain_DCT:
    DCT_model_name = ""
if retrain_REL:
    relation_model_name = ""

token_window = 15
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
                       doc_time_path=DCT_model_name,
                       rel_classifier_path=relation_model_name,
                       greedy=greedy,
                       transitive=transitive)
    if retrain_DCT:
        utils.save_model(bp.doc_time_model, name=DCT_model_name)
    if retrain_REL:
        utils.save_model(bp.annotator.model, name=relation_model_name)
    # Where the magic happens
    bp.predict(test_path)
    print(bp.evaluate(test_path))


if __name__ == "__main__":
    complete_base()
