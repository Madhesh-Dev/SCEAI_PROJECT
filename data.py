
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.data.txt")

df.iloc[:,0:4] = df.iloc[:,0:4].apply(lambda x: x / np.max(x))


X = df.iloc[:,0:4]

y = pd.Series(pd.factorize(df.iloc[:,4])[0])

X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


def getClassArray(classNumber):
    listOfInstancesIndex = y_train.index[y_train==classNumber].tolist()

    return X_train.iloc[listOfInstancesIndex,:]

