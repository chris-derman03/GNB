import pandas as pd
import numpy as np
from naive_bayes import GNB

# How to use the GNB classifier

# Example Training data
#   One df for inputs
#   One df (1 column) for output labels
x_train = pd.read_csv("src/example_data/X-train.csv", header=None) # Filepath for data was unique to my setup. Change as needed.
y_train = pd.read_csv("src/example_data/y-train.csv", header=None)

# Example prediction set (df of inputs)
#    We will predict labels for each of these data points
test = pd.read_csv("src/example_data/X-test.csv", header=None)

# Create and train a GNB model on our training data
model = GNB()
model.fit(x_train,y_train)

# Make predictions on our test set (returns a numpy array)
predictions = model.predict(test)

# Add our predictions as column in the test data set
test['predictions'] = pd.Series(predictions)
print(test)
