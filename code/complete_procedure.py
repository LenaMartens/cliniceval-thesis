import data
import classification
import inference
import output
import utils

DCT_model_name = "SupportVectorMachine1.0"
relation_model_name = "LogisticRegression1.0"


def complete(trainpath, testpath, retrain_dct=True, repredict_dct=True, retrain_rc=True):
    print("Reading the documents")
    if retrain_dct or retrain_rc:
        train_documents = data.read_all(trainpath)
    if repredict_dct:
        test_documents = data.read_all(testpath)

    if repredict_dct:
        if retrain_dct:
            print("Training DCT")
            DCT_model = classification.train_doctime_classifier(train_documents)
            utils.save_model(DCT_model, DCT_model_name)
        else:
            DCT_model = utils.load_model(DCT_model_name)
        print("Predicting DCT")
        predicted_docus = classification.predict_DCT(test_documents, DCT_model)
        for doc in predicted_docus:
            utils.save_document(doc, doc.id)

    if retrain_rc:
        print("Training pre-inference classifier")
        relation_model = classification.train_relation_classifier(train_documents)
        utils.save_model(relation_model, relation_model_name)
    else:
        relation_model = utils.load_model(relation_model_name)

    print("Inferring document relations")
    # Also outputs the document
    inference.infer_relations_on_documents(utils.document_generator(), relation_model)


if __name__ == "__main__":
    complete(utils.train, utils.dev, repredict_dct=True, retrain_rc=True)
