import pandas as pd
from naive_bayes import GNB

# Example Training data
#   One df for inputs
#   One df (1 column) for output labels
x_train = pd.read_csv("src/example_data/X-train.csv", header=None)
y_train = pd.read_csv("src/example_data/y-train.csv", header=None)

# Example prediction set (df of inputs)
#    We will predict labels for each of these data points
test = pd.read_csv("src/example_data/X-test.csv")

model = GNB()
print(model.fit(x_train,y_train))