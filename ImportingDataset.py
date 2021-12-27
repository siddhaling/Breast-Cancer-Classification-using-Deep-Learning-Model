!pip install tensorflow-gpu==2.3.0-rc0

#Importing libraries and supporting libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt

import pandas.util.testing as tm
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loading the Dataset
cancer = datasets.load_breast_cancer()
print(cancer.DESCR

X = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
X.head()
y = cancer.target
y
cancer.target_names
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
X_train.shape
X_test.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = X_train.reshape(455,30,1)
X_test = X_test.reshape(114, 30, 1)
