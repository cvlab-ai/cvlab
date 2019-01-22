import numpy as np

from cvlab.diagram.elements.base import *


def round_11(data, middle):
    if middle:
        delta = 0.33
    else:
        delta = 0
    return -1 * (data < delta) + 1 * (data >= delta)


def round_05(data, middle, classes_count=2):
    if middle:
        d1, d2 = 0.34, 0.16
    else:
        d1, d2 = 0.5, 0

    ret = data * 0
    for c in range(classes_count - 1):  # todo: dla duzej liczby klas to bedzie meeega wolne...
        ret += c * np.logical_and(data >= c - d1, data < c + d1)
        if middle:
            c += 0.5
            ret += c * np.logical_and(data >= c - d2, data < c + d2)
    ret += (classes_count - 1) * (data >= classes_count - 1 - d1)
    return ret


class Trainable(NormalElement):
    def train(self, train_data, responses, sample_weights):
        pass

    def predict(self, v):
        pass

    def predict_all(self, vvv):
        return [self.predict(v) for v in vvv]

    # TODO: to wszystko tutaj jest DO BANI. Trzeba przedyskutowac i zmienic, bo jest za duzo mozliwosci aktualnie.
    # jachoo: moja sugestia - przerobic WSZYSTKIE klasyfikatory na wielowyjsciowe (czyli zamiast numerow klas zeby byly
    #         zawsze prawdopodobienstwa. To uprosci BARDZO liczenie wynikow procentowych itp.
    #         sa dwa minusy w takim wypadku: utrata spojnosci z bibliotekami i troche wieksze zuzycie pamieci

    # scoring defs
    TYPE_CLASSES = 0  # klasy numerowane od 0
    TYPE_CLASSES_WITH_MIDDLE = 1  # j.w. oraz odpowiedz "cos pomiedzy" (wartosci 0.5, 1.5 itd.)
    TYPE_REAL_CENTER_0 = 2  # wyniki bedace liczbami rzeczywistymi typowo od -1 do 1
    TYPE_REAL_CENTER_0_WITH_MIDDLE = 3  # j.w. oraz odpowiedz "cos pomiedzy" (wartosci bliskie 0)
    TYPE_REAL_CENTER_05 = 4  # wyniki bedace liczbami rzeczywistymi typowo od 0 do 1
    TYPE_REAL_CENTER_05_WITH_MIDDLE = 5  # j.w. oraz odpowiedz "cos pomiedzy" (wartosci bliskie 0.5)

    # cross-validation defs
    CV_SIMPLE = 0  # zbiory sa budowane po kolei (pierwsze n/k elementow do pierwszego zbioru itd.)
    CV_STEPPED = 1  # zbiory sa budowane co k elementow (rownomierny podzial danych z calego zbioru wejsciowego)

    output_type = TYPE_CLASSES
    classes_count = 2
    cv_type = CV_SIMPLE

    # returns: [avg errors], [valid percent]
    def score_all(self, predicted, real):
        assert len(predicted) == len(real)
        sample_count = len(predicted)
        errors = None
        valid = None
        outputs = 0
        for p, r in zip(predicted, real):
            self.may_interrupt()
            assert np.alen(p) == np.alen(r)
            if not outputs:
                outputs = np.alen(p)
                errors = np.zeros(outputs)
                valid = np.zeros(outputs)
            e, v = self.score(p, r)
            errors += e
            valid += v
        errors /= sample_count
        valid /= sample_count
        valid_total = valid.sum() / outputs
        return errors, valid, valid_total, sample_count

    @staticmethod
    def format_scores(errors, valid, valid_total, sample_count):
        return "errors: " + str(errors) + " valid: " + str(valid) + " total: " + str(valid_total) + " samples: " + str(
            sample_count)

    def score(self, predicted, real):
        error = abs(predicted - real)
        if self.output_type == self.TYPE_CLASSES:
            valid = 1 * (round_05(predicted, False, self.classes_count) == round_05(real, False, self.classes_count))
        elif self.output_type == self.TYPE_CLASSES_WITH_MIDDLE:
            valid = 1 * (round_05(predicted, True, self.classes_count) == round_05(real, True, self.classes_count))
        elif self.output_type == self.TYPE_REAL_CENTER_0:
            valid = 1 * (round_11(predicted, False) == round_11(real, False))
        elif self.output_type == self.TYPE_REAL_CENTER_0_WITH_MIDDLE:
            valid = 1 * (round_11(predicted, True) == round_11(real, True))
        elif self.output_type == self.TYPE_REAL_CENTER_05:
            valid = 1 * (round_05(predicted, False, 2) == round_05(real, False, 2))
        elif self.output_type == self.TYPE_REAL_CENTER_05_WITH_MIDDLE:
            valid = 1 * (round_05(predicted, True, 2) == round_05(real, True, 2))
        else:
            raise Exception("Wrong scoring type")
        return error, valid

    def cross_validate(self, train_data, responses, sample_weights, k):
        if k <= 0: raise ValueError("Parameter k must be positive")
        outputs = np.alen(responses[0])
        errors, valid, valid_total, total_count = np.zeros(outputs), np.zeros(outputs), 0, 0
        samples = train_data.shape[0]
        if k == 1: k = samples
        cv_step = samples / float(k)
        cv_actual = 0
        for i in range(k):
            if self.cv_type == Trainable.CV_SIMPLE:
                testing_ids = list(range(int(cv_actual), min(int(cv_actual + cv_step), samples)))
                cv_actual += cv_step
            else:
                testing_ids = list(range(i, samples, k))
            current_data = np.delete(train_data, testing_ids, axis=0)
            current_resp = np.delete(responses, testing_ids, axis=0)
            current_weights = np.delete(sample_weights, i, axis=0)
            self.may_interrupt()
            self.train(current_data, current_resp, current_weights)
            self.may_interrupt()
            testing_data = train_data[testing_ids]
            testing_pred = self.predict_all(testing_data)
            testing_real = responses[testing_ids]
            self.may_interrupt()
            e, v, vt, cnt = self.score_all(testing_pred, testing_real)
            errors += e
            valid += v
            valid_total += vt
            total_count += cnt
        return errors / k, valid / k, valid_total / k, total_count

    def train_and_evaluate(self, train_data, responses, sample_weights, cv_k=1):
        if not cv_k:
            cv_scores = None
        else:
            cv_scores = self.cross_validate(train_data, responses, sample_weights, cv_k)

        # todo: dorobic liste blednych rozpoznan (?)

        self.train(train_data, responses, sample_weights)
        train_score = self.score_all(self.predict_all(train_data), responses)

        return """Accuracy:
train data:\t{}
cross-validation:\t{}""".format(self.format_scores(*train_score),
                                self.format_scores(*cv_scores) if cv_scores else "---")

    def get_sample_weights(self, responses):
        counter = {}
        for class_ in responses:
            if class_ in counter:
                counter[class_] += 1
            else:
                counter[class_] = 1
        samples = len(responses)
        length = len(counter)
        sample_weights_map = {}
        for class_, count in counter.items():
            weight = float(samples)/(length * count)
            sample_weights_map[class_] = weight
        weights = []
        for class_ in responses:
            weight = sample_weights_map[class_]
            weights.append(weight)
        return np.array(weights, np.float32)
