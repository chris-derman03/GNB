import pandas as pd
import numpy as np

# Assume all n features are normally distributed and independent
# For a new data point in R^n, pick the label yi that maximizes
#       P(x1|Y=yi) * P(x2|Y=yi) * ... * P(xn|Y=yi) * P(Y=yi)
class GNB:

    # Output of fit(X,y) 
    # (see end of fit(self,X,y) for details)
    __model = {}

    def __init__(self):
        None

    # Train our model from a set of inputs and output labels
    #   X should be array-like with n features (columns) and d data points (rows)
    #       All data must be numeric. Columns can have any name (no duplicate names)
    #   y should be array-like with 1 columns and d rows
    def fit(self, X, y):

        # Convert inputs to pandas dataframes
        training_input = pd.DataFrame(X).reset_index(drop=True)
        training_labels = pd.DataFrame(y).reset_index(drop=True)

        if (training_input.shape[0] != training_labels.shape[0]):
            raise Exception("X and y should have same number of data points.")
        elif (training_labels.shape[1] != 1):
            raise Exception("Labels should be one column.")
        
        # Combine the input columns and label column
        training_input['label'] = training_labels

        #---------------------------------------------------------
        # Training phase
        #---------------------------------------------------------

        # Get each unique label's proportion in the dataset
        label_props = (training_labels.iloc[:,0].value_counts() / training_labels.shape[0]).sort_index()

        # Initialize our output dictionary
        out = {}

        # For each unique label
        for label in label_props.keys():
            
            # Get all data points with that label
            l_df = training_input[training_input['label']==label].drop(columns=['label'])

            # Initialize inner dictionary for this label
            out_sub = {}

            # Add P(Y=yi) where yi is the current label
            out_sub.update({'label_prop':label_props[label]})

            n = l_df.shape[0]
            # For each columns (each feature) for the data that had this label
            for col in l_df:

                # Get the actual column
                c = l_df[col]
                
                # Calculate MLE for mean and spread
                mu_MLE = np.mean(c)
                std_MLE = np.sqrt(np.sum(((c - mu_MLE)**2)) / n)

                # Add it to the inner dictionary with this column title as the key
                out_sub.update({col:(mu_MLE,std_MLE)})

            # Once we have done the above for each column (feature), the inner dictionary is done
            # Add it to the outer dictionary with this label as key
            out.update({label:out_sub})

        # Output is a python dictionary
        #   One key for each unique label
        #       Each label gets its own python dictionary.
        #           Each label's first key-value pair is its proportion in the data set
        #           The other keys are each feature column title
        #               The values for each feature are (MLE for mean, MLE for standard deviation)
        self.__model = out


    # Calculate the value of the gaussian pdf at the numerical point x
    #    mu - mean of the gaussian pdf
    #    std - standard deviation of the gaussian pdf
    def _gauss(self, x, mu, std):

        # Check if inputs are numerical
        if (type(x) != int and type(x) != float):
            raise Exception("x should be numerical")
        elif (type(mu) != int and type(mu) != float):
            raise Exception("Mean should be numerical")
        elif (type(std) != int and type(std) != float):
            raise Exception("Standard Deviation should be numerical")

        var = std**2 # variance

        # Constant of the gaussian pdf formula
        const = 1 / (np.sqrt(2*np.pi)*std)

        # Exponent of the gaussian pdf formula
        e = ((x - mu)**2) / (-2 * var)

        # Calculate the value and return
        return const * np.exp(e)
