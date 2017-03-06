import random

from sklearn import svm, linear_model
import utils
from data import Relation
from feature import WordVector, TimeRelationVector
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
        print(len(trainingdata))
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
	input = scipy.sparse.csr_matrix(input)
        self.machine = svm.SVC()
        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
        sample = sample.get_vector().reshape(1, -1)
        return self.machine.predict(sample)


def generate_training_data(documents):
    feature_vectors = []
    for document in documents:
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature_vectors.append(WordVector(entity))
    return feature_vectors


def generate_training_candidates(documents):
    feature_vectors = []
    for document in documents:
        # Get positive candidates
        entities = list(document.get_entities())
        relations = document.get_relations()
        for relation in relations:
            feature_vectors.append(TimeRelationVector(relation))
        # Generate negative candidates (as many as there are positive)
        added = 0
        maxr = len(entities)
        relation_len = len(relations)
        added_dict = {}
        while added < relation_len:
            [source_id, target_id] = random.sample(range(0, maxr), 2)
            if (source_id, target_id) not in added_dict:
                source = entities[source_id]
                target = entities[target_id]
                if not document.relation_exists(source, target) and source.paragraph == target.paragraph:
                    relation = Relation(source=source, target=target, positive=False)
                    feature_vectors.append(TimeRelationVector(relation))
                    added += 1
                    added_dict[(source_id, target_id)] = True
    return feature_vectors


def train_doctime_classifier(docs):
    features = generate_training_data(docs)
    svm = SupportVectorMachine(features, "doc_time_rel")
    return svm


def train_relation_classifier(docs):
    features = generate_training_candidates(docs)
    lr = LogisticRegression(features)
    return lr


def predict_DCT(documents, model=None):
    for document in documents:
        for entity in document.get_entities():
            if entity.get_class() == "Event":
                feature = WordVector(entity)
                dct = model.predict(feature)
                entity.doc_time_rel = dct[0]
    return documents


if __name__ == '__main__':
    from classification import SupportVectorMachine, LogisticRegression

    docs = utils.get_documents_from_file(utils.store_path)
    train_doctime_classifier(docs)
    # features = generate_training_candidates(docs)
    # lr = LogisticRegression(features)
    # utils.save_model(lr, name="LogisticRegression_randomcandidate")
    # for i in range(10):
    #     print("predicted: " + str(lr.predict(features[i])) + " actual: " + str(features[i].entity.positive))
