import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random as rand


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
        return classify_with_sign(self.w, x)


# not in class Perceptron:

def classify_with_sign(w, x):
    return np.sign(np.dot(x, w.T))


def calc_d1(m, w):
    sample_pts = np.random.multivariate_normal([0, 0],
                                               np.identity(2), m)
    sample_labels = classify_with_sign(w, sample_pts)

    while np.equal(abs(np.sum(sample_labels)), m):  # if labels are -1/1
        sample_pts = np.random.multivariate_normal([0, 0],
                                                   np.identity(2), m)
        sample_labels = classify_with_sign(w, sample_pts)

    return sample_pts, sample_labels


def calc_d2(m):
    sample_arr = [0] * m
    label_arr = [0] * m

    for i in range(m):
        pnt, label = get_rand_pnt_and_label_from_rect()
        sample_arr[i] = pnt
        label_arr[i] = label

    return sample_arr, label_arr


def get_rand_pnt_and_label_from_rect():
    random_num = rand.randint(0, 1)
    if random_num == 1:
        x = np.random.uniform(-3, 1)
        y = np.random.uniform(1, 3)
        label = 1
    else:
        x = np.random.uniform(-1, 3)
        y = np.random.uniform(-3, -1)
        label = -1
    return np.array([x, y]), label


def get_accuracy(test_labels, predicts):
    return accuracy_score(test_labels, predicts)


def train_d1(perceptron_learner, svm_learner):
    accuracy_percp = []
    accuracy_svm = []

    for m in m_arr:
        sum_accuracy = 0
        sum_accuracy2 = 0
        for i in range(500):
            d1_training, d1_training_label = calc_d1(m, w)
            d1_test, d1_test_label = calc_d1(k, w)

            # train percpetron
            perceptron_learner.fit(d1_training, d1_training_label)  # X,y
            percep_test_label = perceptron_learner.predict(d1_test)

            # train svm
            svm_learner.fit(d1_training, d1_training_label)
            svm_test_label = svm_learner.predict(d1_test)

            sum_accuracy += get_accuracy(percep_test_label,
                                         d1_test_label)
            sum_accuracy2 += get_accuracy(svm_test_label,
                                          d1_test_label)

        accuracy_percp.append(sum_accuracy / 500)
        accuracy_svm.append(sum_accuracy2 / 500)

    return accuracy_percp, accuracy_svm


def train_d2(perceptron_learner, svm_learner):
    accuracy_percp = []
    accuracy_svm = []

    for m in m_arr:
        sum_accuracy = 0
        sum_accuracy2 = 0
        for i in range(500):
            d2_training, d2_training_label = calc_d2(m)
            while np.equal(abs(np.sum(d2_training_label)), m):
                d2_training, d2_training_label = calc_d2(m)
            d2_test, d2_test_label = calc_d2(k)

            # train percpetron
            perceptron_learner.fit(d2_training, d2_training_label)  # X,y
            percep_test_label = perceptron_learner.predict(d2_test)

            # train svm
            svm_learner.fit(d2_training, d2_training_label)
            svm_test_label = svm_learner.predict(d2_test)

            sum_accuracy += get_accuracy(percep_test_label,
                                         d2_test_label)
            sum_accuracy2 += get_accuracy(svm_test_label,
                                          d2_test_label)

        accuracy_percp.append(sum_accuracy / 500)
        accuracy_svm.append(sum_accuracy2 / 500)

    return accuracy_percp, accuracy_svm


# globals:
k = 10000  # == test size
m_arr = [5, 10, 15, 25, 70]  # == training size
w = np.array([0.3, -0.5])  # w for sign(<w,x>)


def main():
    perceptron_learner = Perceptron()
    svm_learner = svm.SVC(C=1e10, kernel='linear')

    # train d1
    # accuracy_percp, accuracy_svm = train_d1(perceptron_learner, svm_learner)
    # plt.plot(m_arr, accuracy_percp, label="Perceptron")
    # plt.plot(m_arr, accuracy_svm, label="SVM")
    # plt.legend()
    # plt.show()

    # train d2
    accuracy_percp, accuracy_svm = train_d2(perceptron_learner, svm_learner)

    plt.plot(m_arr, accuracy_percp, label="Perceptron")
    plt.plot(m_arr, accuracy_svm, label="SVM")
    plt.legend()
    plt.show()


def perceptron_test1():
    X = [[-2, 4, -1],
         [4, 1, -1],
         [1, 6, -1],
         [2, 4, -1],
         [6, 2, -1]]
    y = [-1, -1, 1, 1, 1]

    a = Perceptron()
    print(a.fit(X, y))


if __name__ == '__main__':  main()
