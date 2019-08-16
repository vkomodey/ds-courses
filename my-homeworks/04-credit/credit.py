#!/usr/bin/env python3.7
# -*- coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FILEPATH = './credit_train_1.csv'
# TODO do one operation
whole_set = pd.read_csv(FILEPATH, index_col='client_id')
train_set = pd.read_csv(FILEPATH, usecols=list(range(14)), index_col='client_id')
target_set = pd.read_csv(FILEPATH, usecols=[14])

X_train, X_test, y_train, y_test = train_test_split(train_set, target_set, test_size=0.3, random_state=42)

y_train = np.ravel(y_train["open_account_flg"].values)
y_test = np.ravel(y_test["open_account_flg"].values)

clf = LogisticRegression(solver='lbfgs', max_iter=200).fit(X_train, y_train);

predictions = clf.predict(X_test)

print(classification_report(y_train, predictions))
print(confusion_matrix(y_train, predictions))
print(sum(predictions))
