import numpy as np
from sklearn import svm


class Perceptron:

    def fit(self, X, y):

        col_len = len(X[0])  # d
        row_len = len(y)     # m

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
    def classify_with_sign(w , x):
        if np.dot(w, x) > 0:
            return 1
        return -1

    @staticmethod
    def calc_d1():
        m_arr = [5, 10, 15, 25, 70]
        w_arr = [0.3, -0.5]

        d1 = np.random.multivariate_normal([0,0], np.identity(2))
        samples = []
        for m in m_arr:
            samples.append()

def main():
    test_size = 1000  # == k

    clf = svm.SVC(C=1e10, kernel='linear')
#    clf.fit()
    pass


def perceptron_test(X, y):
    X = [[-2, 4, -1],
         [4, 1, -1],
         [1, 6, -1],
         [2, 4, -1],
         [6, 2, -1]]
    y = [-1, -1, 1, 1, 1]

    a = Perceptron()
    print(a.fit(X,y))


if __name__ == '__main__':  main()
