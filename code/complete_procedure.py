import data
import classification
import inference
import output
import utils


def complete(corpuspath):
    print("Reading the documents")
    documents = data.read_all(corpuspath)
    print("Training pre-inference classifier")
    relation_model = classification.train_relation_classifier(documents)
    print("Infering document relations")
    infered_docus = inference.infer_relations_on_documents(documents, relation_model)
    print("Outputting results as XMLs")
    output.output_docs_as_xml(infered_docus)


if __name__ == "__main__":
    complete(utils.dev)
