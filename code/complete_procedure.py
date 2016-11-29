import data
import classification
import inference
import output
import utils


def complete(corpuspath):
    documents = data.read_all(corpuspath)
    relation_model = classification.train_relation_classifier(documents)
    infered_docus = inference.infer_relations_on_documents(documents, relation_model)
    output.output_docs_as_xml(infered_docus)


if __name__ == "__main__":
    complete(utils.dev)
