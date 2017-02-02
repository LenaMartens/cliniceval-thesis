import data
import classification
import inference
import output
import utils


def complete(trainpath, testpath):
    print("Reading the documents")
    train_documents = data.read_all(trainpath)
    test_documents = data.read_all(testpath)
    print("Training DCT")
#    DCT_model = classification.train_doctime_classifier(train_documents)
    print("Predicting DCT")
#    predicted_docus = classification.predict_DCT(test_documents, DCT_model)
    
    predicted_docus = utils.get_documents_from_file()
    print(len(predicted_docus))
    print("------------------------------------------")
#    for doc in predicted_docus:	
#	utils.save_document(doc, doc.id)
    print("Training pre-inference classifier")
    relation_model = classification.train_relation_classifier(train_documents)
    print("Infering document relations")
    inference.infer_relations_on_documents(predicted_docus, relation_model)
    print("Outputting document")
   # output.output_doc(predicted_docus)

if __name__ == "__main__":
    complete(utils.dev, utils.test)
