import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    logistic_reg = LogisticRegression()
    df = pd.read_csv("spam.data", sep=" ")

    #for i in range(10):
    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
    y_train_set = train_set.iloc[:, -1]

    #  learn train_set, predict test_set
    logistic_reg.fit(train_set, y_train_set)
    y_test_predicts = np.argsort(logistic_reg.predict_proba(test_set))

    print(y_test_predicts)

    # logistic_reg.fit(train_set)
    # print(logistic_reg.predict_proba(test_set))
    # test_set = df.sample(n= 1000, axis=0)
    # print(test_set)
    # print(test_set.shape)
    # for x in range(1, 100):
    #     rows = np.random.rand(len(df)) < x / 100
    #     train = df[rows]
    #     test = df[~rows]
    pass


if __name__ == '__main__':  main()
