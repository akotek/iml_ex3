import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class Perceptron:
    def fit(self, X, y):

        col_len = len(X[0])  # d
        row_len = len(y)  # m

        self.w = np.zeros(shape=(1, col_len))

        while True:
            exists_i = False
            for i in range(row_len):
                if np.dot(y[i],
                          np.dot(self.w, X[i])) <= 0:
                    exists_i = True
                    self.w += np.dot(y[i], X[i])
                    break
            if exists_i is False:
                return self.w

    def predict(self, x):
        return Distribution.classify_with_sign(self.w, x)


class Distribution:
    @staticmethod
    def classify_with_sign(w, x):
        return np.sign(np.dot(w, x))  # 1 if > 0, else: -1

    @staticmethod
    def calc_d1(m, w):

        label_pts = []
        sample_pts = []
        for i in range(m):
            pnts_from_random = np.random.multivariate_normal([0, 0],
                                                             np.identity(2))
            sample_pts.append(pnts_from_random)
            label_pts.append(
                Distribution.classify_with_sign(w, pnts_from_random))

        return sample_pts, label_pts

    @staticmethod
    def get_accuracy(label_set, true_label_set):

        accuracy_sum = 0
        num_of_labels = len(label_set)
        for i in range(num_of_labels):
            if label_set[i] == true_label_set[i]:
                accuracy_sum += 1

        return accuracy_sum / num_of_labels


def main():
    k = 10000  # == test size
    m_arr = [5, 10, 15, 25, 70]  # == training size
    w = [0.3, -0.5]
    accuracy = []
    sum_accuracy = 0

    for m in m_arr:

        for i in range(500):
            d1_training, d1_training_label = Distribution.calc_d1(m, w)
            d1_test, d1_test_label = Distribution.calc_d1(k, w)

            learner = Perceptron()
            learner.fit(d1_training, d1_training_label)
            learner_test_label = [learner.predict(x) for x in d1_test]

            sum_accuracy += Distribution.get_accuracy(learner_test_label, d1_test_label)

        accuracy.append(sum_accuracy/k)

    plt.plot(m_arr, accuracy, label="Perceptron")
    plt.legend()
    plt.show()


    # print(d1_sample_set)
    # print(d1_training_label)
    clf = svm.SVC(C=1e10, kernel='linear')


#    clf.fit()


# get_result(50)

def percep_test(X, y):
    X = [[-2, 4, -1],
         [4, 1, -1],
         [1, 6, -1],
         [2, 4, -1],
         [6, 2, -1]]
    y = [-1, -1, 1, 1, 1]

    a = Perceptron()
    print(a.fit(X, y))


if __name__ == '__main__':  main()
