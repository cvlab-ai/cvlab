from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from cvlab.diagram.elements.base import *

from .ml import Trainable


class ScikitSimpleTrainable(Trainable):

    def __init__(self):
        super(ScikitSimpleTrainable, self).__init__()
        self.model = None

    def build_classifier(self):
        pass

    def get_attributes(self):
        return [Input("train data"),
                Input("responses")], \
               [Output("model"),
                Output("log")], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        train_data = inputs["train data"].value
        responses = inputs["responses"].value
        sample_weights = self.get_sample_weights(responses)
        log = self.train_and_evaluate(train_data, responses, sample_weights)
        outputs["model"] = Data(self.model)
        outputs["log"] = Data(log)

    def train(self, train_data, responses, sample_weights):
        self.model = self.build_classifier()
        try:
            self.model.fit(train_data, responses, sample_weight=sample_weights)
        except TypeError:
            self.model.fit(train_data, responses)

    def predict(self, v):
        return self.model.predict(v)


class ScikitSimplePrediction(NormalElement):

    name = "Predict"
    comment = "General prediction for scikit classifiers"

    def __init__(self):
        super(ScikitSimplePrediction, self).__init__()
        self.model = None

    def get_attributes(self):
        return [Input("test data"),
                Input("model")], \
               [Output("responses")], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        self.model = inputs["model"].value
        self.test_data = inputs["test data"].value
        predictions = []
        for v in self.test_data:
            result = self.model.predict(v)
            predictions.append(result)
        result = np.array(predictions)
        outputs["responses"] = Data(result)


class NaiveBayesBernoulliTrain(ScikitSimpleTrainable):
    name = "NaiveBayesBernoulliTrain"
    comment = "Bernoulli Naive Bayes classifier training"

    def build_classifier(self):
        return BernoulliNB()


class NaiveBayesGaussTrain(ScikitSimpleTrainable):
    name = "NaiveBayesGaussTrain"
    comment = "Gaussian Naive Bayes classifier training"

    def build_classifier(self):
        return GaussianNB()


class DecisionTreeTrain(ScikitSimpleTrainable):
    name = "DecisionTreeTrain"
    comment = "Decision Tree classifier training"

    def __init__(self):
        super(DecisionTreeTrain, self).__init__()
        self.model = None
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1

    def build_classifier(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )

    def get_attributes(self):
        return [Input("train data"),
                Input("responses")], \
               [Output("model"),
                Output("log")], \
               [
                   IntParameter("max_depth", value=0, min_=0),
                   IntParameter("min_samples_split", value=2, min_=0),
                   IntParameter("min_samples_leaf", value=1, min_=0)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        train_data = inputs["train data"].value
        responses = inputs["responses"].value
        self.max_depth = parameters["max_depth"] if parameters["max_depth"] != 0 else None
        self.min_samples_split = parameters["min_samples_split"]
        self.min_samples_leaf = parameters["min_samples_leaf"]
        sample_weights = self.get_sample_weights(responses)
        log = self.train_and_evaluate(train_data, responses, sample_weights)
        outputs["model"] = Data(self.model)
        outputs["log"] = Data(log)

    def train(self, train_data, responses, sample_weights):
        self.model = self.build_classifier()
        self.model.fit(train_data, responses, sample_weight=sample_weights)


class RandomForestTrain(ScikitSimpleTrainable):
    name = "RandomForestTrain"
    comment = "Random Forest classifier training"

    def build_classifier(self):
        return RandomForestClassifier()


register_elements_auto(__name__, locals(), "Machine learning - scikit", 12)

