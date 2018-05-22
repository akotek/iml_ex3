import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def main():
    logistic_reg = LogisticRegression()
    df = pd.read_csv("spam.data", sep=" ")

    #for i in range(10):
    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
    train_set_label = train_set.iloc[:, -1]
    test_set_label = test_set.iloc[:, -1]

    #  learn train_set, predict test_set
    logistic_reg.fit(train_set, train_set_label)
    test_set_predicts = logistic_reg.predict_proba(test_set)
    sorted_arr = np.sort(test_set_predicts, axis=0)

    NP = test_set_label.sum()  # all labels which are 1 == sum

    # init both arrays
    Ni = np.zeros(NP+1) # Ni == TPR
    FPR = np.zeros(NP+1)

    print(test_set_label)
  #  for i in range(NP):




    # plt.plot(FPR, Ni, label="ROC CURVE")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':  main()
