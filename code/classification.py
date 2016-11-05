from sklearn import svm
import utils
from feature import WordFeatureVector


class Classifier:
    def train(self, trainingdata):
        pass

    def predict(self, sample):
        pass

    def __init__(self, trainingdata, class_to_fy):
        # List of FeatureVectors
        self.class_to_fy = class_to_fy
        self.train(trainingdata)


class SupportVectorMachine(Classifier):
    def train(self, trainingdata):
        input = [x.vector for x in trainingdata]
        output = [getattr(x.entity, self.class_to_fy) for x in trainingdata]
        self.machine = svm.SVC()
        print('started training')
        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
	sample = sample.vector.reshape(1, -1)
        print(sample)
	return self.machine.predict(sample)


def generate_training_data(documents):
    feature_vectors = []
    for document in documents:
        print(document)
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature_vectors.append(WordFeatureVector(entity))
    return feature_vectors


if __name__ == '__main__':
    from classification import SupportVectorMachine
    docs = utils.get_documents_from_file(utils.store_path)
    features = generate_training_data(docs)
    '''
    sv = SupportVectorMachine(features, "doc_time_rel")
    utils.save_model(sv, name="SupportVectorMachine_dev")
    '''
    sv = utils.load_model("SupportVectorMachine_dev")
    for i in range(10):
        print("predicted: " + str(sv.predict(features[i])) + " actual: " + str(features[i].entity.doc_time_rel))
