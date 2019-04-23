from cvlab.diagram.elements.base import *

from .ml import Trainable


class SvmTrain(Trainable):
    name = "SvmTrain"
    comment = "Support Vector Machine classifier training"

    output_type = Trainable.TYPE_CLASSES
    classes_count = 2

    def __init__(self):
        super(SvmTrain, self).__init__()
        self.svm = None
        self.train_data = None
        self.params_ = None

    def get_attributes(self):
        return [Input("train data"),
                Input("responses")], \
               [Output("model"),
                Output("log")], \
               [
                   ComboboxParameter("svm_type", values=OrderedDict([
                       ("C_SVC", cv.SVM_C_SVC),
                       ("NU_SVC", cv.SVM_NU_SVC),
                       ("ONE_CLASS", cv.SVM_ONE_CLASS),
                       ("EPS_SVR", cv.SVM_EPS_SVR),
                       ("NU_SVR", cv.SVM_NU_SVR)
                    ])),
                   ComboboxParameter("kernel_type", values=OrderedDict([
                       ("LINEAR", cv.SVM_LINEAR),
                       ("POLY", cv.SVM_POLY),
                       ("RBF", cv.SVM_RBF),
                       ("SIGMOID", cv.SVM_SIGMOID)
                   ]), default_value_idx=2),
                   FloatParameter("degree", value=0.000001),
                   FloatParameter("gamma", value=1.0),
                   FloatParameter("coef0", value=0.0),
                   FloatParameter("C", value=1.0),
                   FloatParameter("nu", value=0.000001),
                   FloatParameter("p", value=0.0),
                   IntParameter("max_iter", value=1000),
                   FloatParameter("epsilon", value=0.000001, min_=0.0000001, max_=1.0),
                   IntParameter("CV_k", value=1, min_=0, max_=100)
            ]

    def process_inputs(self, inputs, outputs, parameters):
        train_data = inputs["train data"].value
        responses = inputs["responses"].value
        print("SVM process_intputs", train_data.shape, responses.shape)
        self.params_ = parameters
        sample_weights = self.get_sample_weights(responses)
        log = self.train_and_evaluate(train_data, responses, sample_weights, parameters["CV_k"])
        outputs["model"] = Data(self.svm)
        outputs["log"] = Data(log)

    def train(self, train_data, responses, sample_weights):
        print("Training SVM with", train_data.shape[0], "samples...")
        params = self.params_
        term_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, params["max_iter"], params["epsilon"])
        svm_params = dict(
            svm_type=params["svm_type"],
            kernel_type=params["kernel_type"],
            degree=params["degree"],
            gamma=params["gamma"],
            coef0=params["coef0"],
            C=params["C"],
            nu=params["nu"],
            p=params["p"],
            term_crit=term_criteria)

        self.svm = cv.SVM()
        self.svm.train(train_data, responses, params=svm_params)

    def predict(self, v):
        return self.svm.predict(v)

    def evaluate_training(self, train_data, responses):
        ok_count = 0
        i = 0
        for v in train_data:
            if self.svm.predict(v) == responses[i]:
                ok_count += 1
            i += 1
        return "Train data result:\n   acc:\t" + str(float(ok_count) / i) + " (" + str(ok_count) + "/" + str(i) + ")"


class SvmTrainNoParams(Trainable):
    name = "SvmTrainNoParams"
    comment = "Support Vector Machine classifier"

    output_type = Trainable.TYPE_CLASSES
    classes_count = 2

    def __init__(self):
        super(SvmTrainNoParams, self).__init__()
        self.svm = None

    def get_attributes(self):
        return [Input("train data"),
                Input("responses")], \
               [Output("model"),
                Output("log")], \
               [IntParameter("CV_k", value=1, min_=0, max_=100)]

    def process_inputs(self, inputs, outputs, parameters):
        train_data = inputs["train data"].value
        responses = inputs["responses"].value
        sample_weights = self.get_sample_weights(responses)
        log = self.train_and_evaluate(train_data, responses, sample_weights, parameters["CV_k"])
        outputs["model"] = Data(self.svm)
        outputs["log"] = Data(log)

    def train(self, train_data, responses, sample_weights=None):
        self.svm = cv.SVM(train_data, responses)

    def predict(self, v):
        return self.svm.predict(v)


class SvmPredict(FunctionGuiElement, ThreadedElement):
    name = "SvmPredict"
    comment = "Support Vector Machine classifier prediction"

    def __init__(self):
        super(SvmPredict, self).__init__()
        self.svm = None

    def get_attributes(self):
        return [Input("test data"),
                Input("model")], \
               [Output("responses")], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        self.svm = inputs["model"].value
        test_data = inputs["test data"].value
        predictions = []
        for v in test_data:
            result = self.svm.predict(v)
            predictions.append(result)
        result = np.array(predictions)
        outputs["responses"] = Data(result)


class AnnMlpTrain(Trainable):
    name = "ANN MLP Train"
    comment = "Artificial Neural Network - Multi layer perceptron - classifier training"

    output_type = Trainable.TYPE_REAL_CENTER_0_WITH_MIDDLE
    classes_count = 2

    def __init__(self):
        super(AnnMlpTrain, self).__init__()
        self.ann = None
        self.train_data = None
        self.params_ = None

    def get_attributes(self):
        return [Input("train data"),
                Input("responses")], \
               [Output("model"),
                Output("log")], \
               [
                   TextParameter("layer_sizes", value="5 3 1", window_title="Neural network layer sizes editor",
                                 window_content="Layers sizes from input to output, separated by spaces:"),
                   ComboboxParameter("activate_func", values=OrderedDict([
                       ("IDENTITY", cv.ml.ANN_MLP_IDENTITY),
                       ("SIGMOID_SYM", cv.ml.ANN_MLP_SIGMOID_SYM),
                       ("GAUSSIAN", cv.ml.ANN_MLP_GAUSSIAN),
                   ]), default_value_idx=1),
                   ComboboxParameter("train_method", values=OrderedDict([
                       ("BACKPROP", cv.ml.ANN_MLP_BACKPROP),
                       ("RPROP", cv.ml.ANN_MLP_RPROP),
                   ]), default_value_idx=1),
                   ComboboxParameter("scaling", values=OrderedDict([
                       ("FULL SCALING", 0),
                       ("NO INPUT SCALE", cv.ml.ANN_MLP_NO_INPUT_SCALE),
                       ("NO OUTPUT SCALE", cv.ml.ANN_MLP_NO_OUTPUT_SCALE),
                       ("NO SCALING", cv.ml.ANN_MLP_NO_INPUT_SCALE + cv.ml.ANN_MLP_NO_OUTPUT_SCALE),
                   ])),
                   # term criteria
                   IntParameter("max_iter", value=10, min_=1, max_=10000),
                   FloatParameter("epsilon", value=0.01, min_=0, max_=1),
                   # backprop params:
                   FloatParameter("dw_scale", value=0.1, min_=0, max_=1),
                   FloatParameter("moment_scale", value=0.1, min_=0, max_=2),
                   # rprop params
                   FloatParameter("dw0", value=0.1),
                   FloatParameter("dw_min", value=0.000001, min_=0),
                   # cross validation
                   IntParameter("CV_k", value=1, min_=0, max_=100),
                   # retraining
                   ComboboxParameter("train_mode", values=OrderedDict([
                       ("RETRAIN", 0),
                       ("UPDATE", cv.ml.ANN_MLP_UPDATE_WEIGHTS)
                   ])),
                   ButtonParameter("retrain_button", self.retrain, "Retrain ANN"),
                   # saving/loading
                   ButtonParameter("save", self.save, "Save net to file"),
               ]

    def process_inputs(self, inputs, outputs, parameters):
        train_data = inputs["train data"].value
        responses = inputs["responses"].value
        # print("ANN process_intputs", train_data.shape, responses.shape)
        self.params_ = parameters
        # sample_weights = self.get_sample_weights(responses)  # todo: zrobic to - musimy uwzglednic, ze ANN zwraca floaty!
        sample_weights = None
        log = self.train_and_evaluate(train_data, responses, sample_weights, parameters["CV_k"])
        outputs["model"] = Data(self.ann)
        outputs["log"] = Data(log)

    def train(self, train_data, responses, sample_weights):
        params = self.params_
        term_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, params["max_iter"], params["epsilon"])
        layers = np.array([train_data.shape[1]] + [int(i) for i in params["layer_sizes"].split()] + [
            responses.shape[1] if len(responses.shape) > 1 else 1])
        flags = 0 + params["scaling"] + params["train_mode"]
        print("Training ANN with", train_data.shape[0], "samples... Layers:", layers)
        if self.ann is None or params["train_mode"] == 0:
            self.ann = cv.ml.ANN_MLP_create()
    
            self.ann.setLayerSizes(layers)
            self.ann.setTrainMethod(params['train_method'])
            self.ann.setBackpropMomentumScale(params['moment_scale'])
            self.ann.setBackpropWeightScale(params['dw_scale'])
            self.ann.setRpropDW0(params['dw0'])
            self.ann.setRpropDWMin(params['dw_min'])
            self.ann.setTermCriteria(term_criteria)
            self.ann.setActivationFunction(params['activate_func'])

            flags -= params['train_mode']

        data = cv.ml.TrainData_create(train_data, cv.ml.ROW_SAMPLE, responses)
        iters = self.ann.train(data, flags)
        print("Done training ANN. Iterations:", iters)

    def predict(self, v):
        ret = self.ann.predict(v.reshape((1, len(v))))[1][0]
        return np.nan_to_num(ret)  # opencv ANN sometimes returns NaNs... ehhh... let's convert them to 0's

    def retrain(self):
        self.ann = None
        self.recalculate(True, False, True)

        # def evaluate_training(self, train_data, responses):
        #     errors = np.zeros(responses.shape[1])
        #     for v, response in zip(train_data, responses):
        #         output = self.ann.predict(v)
        #         errors += abs(response - output)
        #     errors /= train_data.shape[0]
        #     return "Train data result: average errors={}".format(errors)

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save neural network")
        if not path: return
        self.ann.save(str(path))


register_elements("Machine Learning - OpenCV", [SvmTrain, SvmTrainNoParams, SvmPredict, AnnMlpTrain], 20)
