# -*- coding: utf-8 -*-
from tensorflow.keras import *
from keras import *
import numpy as np

# The data represents individuals with the following attributes:
# [Age, Years of study, Years of work, Gender (0 = Male, 1 = Female), Salary]
data = [
        [18, 2, 0, 0, 2100],
        [32, 4, 10, 0, 3400],
        [44, 2, 20, 0, 3200],
        [60, 6, 34, 0, 5500],
        [20, 2, 2, 1, 2300],
        [24, 4, 0, 1, 2700],
        [44, 5, 21, 1, 3200],
        [52, 2, 32, 1, 3200],
        ]

data_array = np.array(data)

# x_train contains the input features from the dataset: selecting the first 8 rows and the first 4 columns (Age, Years of study, Years of work, Gender)
x_train = data_array[0:8, 0:4]

# y_train contains the target variable (Salary) from the dataset: selecting the first 8 rows and the 5th column
y_train = data_array[0:8, 4]


model = models.Sequential()

# Input layer: accepts four input features (Age, Years of study, Years of work, Gender)
model.add(layers.Dense(units=16, input_shape=[4]))

# Intermediate layers: process and transform the data through multiple hidden units
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))

# Output layer: produces a single output value (Salary)
model.add(layers.Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on the training data for 1000 epochs
model.fit(x=x_train, y=y_train, epochs=5000)

gabriel = np.array([[18, 1, 1, 0]])
louise = np.array([[52, 7, 20, 1]])

matteo = np.array([[30, 3, 7, 0]])
nathalie = np.array([[30, 3, 7, 1]])

print(model.predict(gabriel))
print(model.predict(louise))

print(model.predict(matteo))
print(model.predict(nathalie))