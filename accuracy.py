from sklearn.model_selection import train_test_split
from naive_bayes import GNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In this file we will analyze the accuracy of some prediction based on the data in the example_data folder

# Keep in mind GNB will not achieve a very high accuracy on this data. The features in this data are NOT
# normal and are NOT all independent. This data is meant to demonstrate how GNB can still work decently when we 
# make the naive assumption of indepdence and gaussian-ness when doing so was a bad choice.

# Example Training data
#   One df for inputs
#   One df (1 column) for output labels
train_inputs = pd.read_csv("src/example_data/X-train.csv", header=None) # Filepath for data was unique to my setup. Change as needed.
train_labels = pd.read_csv("src/example_data/y-train.csv", header=None)

# Make a train-test split, make predictions and find accuracy
def trial():

    # Make a training-testing split
    x_train, x_test, y_train, y_test = train_test_split(train_inputs , train_labels, test_size=0.25)
    correct_labels = y_test.iloc[:,0].reset_index(drop=True)

    # Create and train a GNB model on our training data
    model = GNB()
    model.fit(x_train,y_train)

    # Make predictions on the test set
    predictions = pd.Series(model.predict(x_test))

    # Compare predictions with test labels to get accuracy of this trial
    return np.mean(predictions == correct_labels)

# To account for randomness of the universe, we will run the above trial and plot a histogram to see how our predictions perfrom ON AVERAGE
accuracies = np.array([])
for _ in range(100):
    acc = trial()
    accuracies = np.append(accuracies, acc)

plt.hist(accuracies)
plt.show() # See the attached .jpg in this repository for the histogram
