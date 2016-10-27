from sklearn import svm


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
        output = [getattr(x, self.class_to_fy) for x in self.trainingdata]
        self.machine = svm.SVC()
        self.machine.fit(input, output)

    def predict(self, sample):
        # sample = FeatureVector
        return self.machine.predict(sample)

    def __init__(self, trainingdata, class_to_fy):
        # List of FeatureVectors
        self.trainingdata = trainingdata
        self.class_to_fy = class_to_fy


def generate_training_data():
    
