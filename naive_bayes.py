import pandas as pd
import numpy as np

class GNB:

    x = 0

    def __init__(self):
        None

    def fit(self, X, y):

        # Convert inputs to pandas dataframes
        training_input = pd.DataFrame(X)
        training_labels = pd.DataFrame(y)

        if (training_input.shape[0] != training_labels.shape[0]):
            raise Exception("X and y should have same number of data points.")
        elif (training_labels.shape[1] != 1):
            raise Exception("Labels should be one column.")

    def print_value(self,):
        print(self.x)