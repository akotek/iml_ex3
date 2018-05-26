import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def getNi(i, test_pred, sorted_test_pred):
    counter = 0
    index = 0
    threshold = 0

    while counter != i:
        if test_pred[sorted_test_pred[index][1] == 1]:
            counter +=1
        threshold +=1
        index +=1
    return threshold



def main():
    logistic_reg = LogisticRegression()
    df = pd.read_csv("spam.data", sep=" ", header=None)

    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
    train_set_label = train_set.iloc[:, -1].values  # DataFrame > Series > Ndarray
    test_set_label = test_set.iloc[:, -1].values

    # remove label column
    train_set = train_set.drop(train_set.columns[57], axis=1)
    test_set = test_set.drop(test_set.columns[57], axis=1)

    # learn train_set, predict test_set
    logistic_reg.fit(train_set, train_set_label)
    test_set_predicts = logistic_reg.predict_proba(test_set)
    test_pred_sorted = np.sort(test_set_predicts, axis=1)

    NP = test_set_label.sum()  # np is POSITIVE
    NN = len(test_set) - NP    # nn is NEGATIVE
    # init both arrays
    TPR_arr = np.zeros(NP+1)
    FPR_arr = np.zeros(NP+1)

#num of samples to take to get i of 1's'

    # for i in range(1):
    #     tpr = i / NP
    #     Ni = getNi(i, test_set_predicts, test_pred_sorted)
    #     fpr = (Ni - i) / NN
    #     TPR_arr[i] = tpr
    #     FPR_arr[i] = fpr
    # TPR_arr[NP] = 1
    # FPR_arr[NP] = 1

    # make average of num repetition
    TPR_arr = TPR_arr/10
    FPR_arr = FPR_arr/10
    # plt.plot(FPR_arr, TPR_arr, label="ROC CURVE")
    # plt.legend()
    # plt.show()

# def calc_roc_curve():
#     repetitions = 10
#     prob_pred, test_pred = get_probabilty_pred()
#     positives = np.sum(test_pred == 1)
#     negatives = test_set_size - positives
#     cum_roc = np.array([[0., 0.] * (positives + 2)]).reshape((positives + 2, 2))
#     for j in range(repetitions):
#         roc_curve = [[0., 0.]]
#         for i in range(positives):
#             tpr = i / positives
#             fpr = (getNi(i, test_pred, prob_pred) - i) / negatives
#             roc_curve.append([tpr, fpr])
#         roc_curve.append([1, 1])
#         cum_roc += np.array(roc_curve)
#     return cum_roc / repetitions
  #  for i in range(NP):




if __name__ == '__main__':  main()
