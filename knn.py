import pandas as pd
from sklearn.model_selection import train_test_split

class knn:
    def __init__(self, k):
        self.k = k


    def fit(self, X, y):
        pass

    def predict(self, x):
        pass

def main():
    df = pd.read_csv("spam.data", sep=" ")

    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
    train_set_label = train_set.iloc[:, -1].values  # DataFrame > Series > Ndarray
    test_set_label = test_set.iloc[:, -1].values

    # remove label column
    train_set = train_set.drop(train_set.columns[57], axis=1)
    test_set = test_set.drop(test_set.columns[57], axis=1)

    arr = [1,2,5,10, 100]

    #for k in arr:
    k = arr[0]
    knn_learner = knn(k)

    knn.fit(train_set, train_set_label)
    test_predicts = knn.predict(test_set_label)



    train_set, test_set = train_test_split(df, test_size=1000 / df.shape[0])
if __name__ == '__main__':  main()
