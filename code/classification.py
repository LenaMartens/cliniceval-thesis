from sklearn import svm, linear_model
import utils
from candidate_generation import generate_training_data, generate_constrained_candidates
from data import read_all
from feature import WordVectorWithContext
import scipy.sparse


class Classifier:
    def train(self, trainingdata):
        pass

    def predict(self, sample):
        pass

    def __init__(self, trainingdata, class_to_fy):
        # List of FeatureVectors
        self.class_to_fy = class_to_fy
        self.train(trainingdata)


class LogisticRegression(Classifier):
    def train(self, trainingdata):
        input = [x.get_vector() for x in trainingdata]
        output = [getattr(x.entity, self.class_to_fy) for x in trainingdata]
        self.machine = linear_model.LogisticRegression()
        self.machine.fit(input, output)

    def predict(self, sample):
        # returns a log probability distribution
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict_proba(sample)

    def __init__(self, trainingdata):
        # List of FeatureVectors
        Classifier.__init__(self, trainingdata, "positive")


class SupportVectorMachine(Classifier):
    def train(self, trainingdata):
        input = [x.get_vector() for x in trainingdata]
        output = [getattr(x.entity, self.class_to_fy) for x in trainingdata]
        print(output)
        input = scipy.sparse.csr_matrix(input)

        # BALANCED BECAUSE OF DATA BIAS + linear
        self.machine = svm.LinearSVC(class_weight='balanced')

        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict(sample)


def train_doctime_classifier(docs):
    features = generate_training_data(docs)
    svm = SupportVectorMachine(features, "doc_time_rel")
    return svm


def train_relation_classifier(docs, token_window):
    features = generate_constrained_candidates(docs, token_window)
    lr = LogisticRegression(features)
    return lr


def predict_DCT(documents, model=None):
    for document in documents:
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature = WordVectorWithContext(entity, document)
                dct = model.predict(feature)
                entity.doc_time_rel = dct[0]
    return documents


if __name__ == '__main__':
    # because of pickle issues
    from classification import SupportVectorMachine, LogisticRegression

    docs = read_all(utils.dev)
    train_doctime_classifier(docs)
# features = generate_training_candidates(docs)
# lr = LogisticRegression(features)
# utils.save_model(lr, name="LogisticRegression_randomcandidate")
# for i in range(10):
#     print("predicted: " + str(lr.predict(features[i])) + " actual: " + str(features[i].entity.positive))
