from sklearn import svm
import utils
from feature import WordFeatureVector


class Classifier:
    trainingdata = []
    class_to_fy = ""
    machine = None

    def train(self):
        pass

    def predict(self, sample):
        pass


class SupportVectorMachine(Classifier):
    def train(self):
        input = [x.vector for x in self.trainingdata]
        output = [getattr(x.entity, self.class_to_fy) for x in self.trainingdata]
        self.machine = svm.SVC()
        print('started training')
        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
        return self.machine.predict(sample.vector)

    def __init__(self, trainingdata, class_to_fy):
        # List of FeatureVectors
        self.trainingdata = trainingdata
        self.class_to_fy = class_to_fy


def generate_training_data(documents):
    feature_vectors = []
    for document in documents:
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature_vectors.append(WordFeatureVector(entity))
    return feature_vectors

if __name__ == '__main__':
    from classification import SupportVectorMachine
    docs = utils.get_documents_from_file(utils.store_path)
    features = generate_training_data(docs)
    sv = SupportVectorMachine(features, "doc_time_rel")
    utils.save_model(sv)
    for i in range(10):
        print("predicted: "+sv.predict(features[i])+" actual: "+features[i].entity.doc_time_rel)