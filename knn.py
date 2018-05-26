import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class knn:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        # compute euclid dist to every training example xi (D(x,xi))
        # ord=2, axis=1 is l2 norm
        distance_arr = np.linalg.norm(self.X - x, axis=1, ord=2)
        # sort by smallest distance, argsort returns Indexes
        closest_pts_inexes = np.argsort(distance_arr, axis=0)[0:self.k]
        # now we select k closest xi's LABELS
        labels_arr = self.y[closest_pts_inexes]
        # and we return the labeling (0 or 1 depends on how much there is)
        return np.argmax(np.bincount(labels_arr))


def main():
    df = pd.read_csv("spam.data", sep=" ", header=None)

    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
    train_set_label = train_set.iloc[:, -1].values  # ndarray
    test_set_label = test_set.iloc[:, -1].values

    # remove label column and make all nd_array

    train_set = train_set.drop(train_set.columns[57], axis=1).values
    test_set = test_set.drop(test_set.columns[57], axis=1).values

    K = [1, 2, 5, 10, 100]
    test_error = np.zeros(len(K))
    counter = 0

    for k in K:
        clf = knn(k)
        # train and predict on test_set
        clf.fit(train_set, train_set_label)
        test_predicts = np.apply_along_axis(clf.predict, axis=1, arr=test_set)
        test_error[counter] = 1 - accuracy_score(test_set_label, test_predicts)
        counter += 1

    plt.plot(K, test_error, label="KNN")
    plt.legend()
    plt.show()

if __name__ == '__main__':  main()