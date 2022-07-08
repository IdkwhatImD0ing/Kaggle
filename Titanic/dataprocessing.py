import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

seed = 1447


def getDataset():
    train = pd.read_csv("train.csv", index_col="PassengerId")
    test = pd.read_csv("test.csv", index_col="PassengerId")

    train, test = manipulateDataset(train, test)

    y_train = train["Survived"]
    x_train = train.drop("Survived", axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=1)

    return x_train, y_train, x_val, y_val, test


def manipulateDataset(train, test):
    train["Cabin"] = train["Cabin"].apply(
        lambda x: x[0] if pd.notna(x) else "na"
    )  #We don't care about room numbers, only about the general cabin they are in. (Eg A, B, C, etc)
    test["Cabin"] = test["Cabin"].apply(lambda x: x[0]
                                        if pd.notna(x) else "na")

    #ID, Name, and Ticket Number logically have no effect on whether passengers survived or not
    train.drop(["Name", "Ticket"], axis=1, inplace=True)
    test.drop(["Name", "Ticket"], axis=1, inplace=True)

    #If passenger age is unknown, use the mean
    train["Age"].fillna(train["Age"].mean(skipna=True), inplace=True)
    test["Age"].fillna(test["Age"].mean(skipna=True), inplace=True)

    #If embarked location is unknown, use the most embarked location
    train["Embarked"].fillna("S", inplace=True)

    #Train does not have any missing fare vaules, as for test, use the average fare value.
    test["Fare"].fillna(test["Fare"].mean(skipna=True), inplace=True)

    #Label encoding
    sex = {'male': 0, 'female': 1}
    train["Sex"] = [sex[i] for i in train["Sex"]]
    test["Sex"] = [sex[i] for i in test["Sex"]]

    embarked = {'S': 0, 'C': 1, 'Q': 2}
    train["Embarked"] = [embarked[i] for i in train["Embarked"]]
    test["Embarked"] = [embarked[i] for i in test["Embarked"]]

    cabin_plot = train[["Cabin", "Survived"]]
    train["Cabin"] = LabelEncoder().fit_transform(train["Cabin"])
    test["Cabin"] = LabelEncoder().fit_transform(test["Cabin"])

    return train, test


def data_statistics():
    x_train, y_train, x_val, y_val, test = getDataset()
    print("\nTrain null values\n")
    print(x_train.isna().sum())
    print("\nTest null values\n")
    print(test.isna().sum())

    print("\nDatatypes:\n")
    print(x_train.info())

    print("\nTrain Distribution:\n")
    print(x_train.describe())
    return


data_statistics()