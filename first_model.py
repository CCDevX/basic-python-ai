# -*- coding: utf-8 -*-

from tensorflow.keras import *
from keras import *
import numpy as np

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([3, 6, 9, 12, 15])

model = models.Sequential()


# Input layer: accepts a single input (input_shape=[1])
model.add(layers.Dense(units=3, input_shape=[1]))

# Intermediate layer: processes and transforms the data
model.add(layers.Dense(units=2))

# Output layer: produces a single output value
model.add(layers.Dense(units=1))

# Compile the model using the Adam optimizer or SGD (Stochastic Gradient Descent)
# Adam is often preferred for faster convergence, while SGD is simpler and may work better in some cases.
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on the training data for 1000 epochs
model.fit(x=x_train, y=y_train, epochs=5000)


print(model.predict(np.array([6])))
print(model.predict(np.array([7])))
print(model.predict(np.array([8])))

print(model.predict(np.array([6])))
print(model.predict(np.array([7])))
print(model.predict(np.array([8])))