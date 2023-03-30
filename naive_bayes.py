import pandas as pd
import numpy as np

# Assume all n features are normally distributed and independent
# For a new data point in R^n, pick the label yi that maximizes
#       P(x1|Y=yi) * P(x2|Y=yi) * ... * P(xn|Y=yi) * P(Y=yi)
class GNB:

    # Output of fit(X,y) 
    # (see end of fit(self,X,y) for details)
    __model = {}

    # Number of features in this model
    n_features = 0

    def __init__(self):
        None

    # Given a pandas dataframe with m columns
    #   Return a new dataframe with columns labeled 0 through m
    def _reset_names(self,df):
        m = df.shape[1]
        return df.rename(columns={x:y for x,y in zip(df.columns,range(0,m))})

    # Train our model from a set of inputs and output labels
    #   X should be array-like with n features (columns) and d data points (rows)
    #       All data must be numeric. Columns can have any name (no duplicate names)
    #   y should be array-like with 1 columns and d rows
    def fit(self, X, y):

        # Convert inputs to pandas DataFrames
        training_input = pd.DataFrame(X).reset_index(drop=True)
        training_labels = pd.DataFrame(y).reset_index(drop=True)

        # Reset column names
        training_input = self._reset_names(training_input)

        self.n_features = training_input.shape[1]

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
        try:
            x = float(x)
        except:
            raise Exception("x should be numerical")
        try:
            mu = float(mu)
        except:
            raise Exception("mu should be numerical")
        try:
            std = float(std)
        except:
            raise Exception("std should be numerical")

        var = std**2 # variance

        # Constant of the gaussian pdf formula
        const = 1 / (np.sqrt(2*np.pi)*std)

        # Exponent of the gaussian pdf formula
        e = ((x - mu)**2) / (-2 * var)

        # Calculate the value and return
        return const * np.exp(e)
    
    # Call fit(X,y) first
    # Given one data point in R^n_features, and the model being trained on some data, return a label prediction
    # vector is array-like and MUST have the same feature column titles as the training data
    def __predict_sub(self, vector):

        # This dict will have a key for each label
        # Each label's value is the GNB likelihood for that label given the input vector
        label_likelihoods = {} # The label with the highest likelihood is our prediction

        # Make sure input and __model are of the correct form
        if ((len(self.__model[0]) - 1) != self.n_features):
            raise Exception("Make sure you called fit(X,y) on the right data!")
        elif (vector.shape[0] != self.n_features):
            raise Exception("Input vector should have data for " + str(self.n_features) + " features.")
        
        # For each label in the model
        for label in self.__model:

            # Get the dictionary for this label
            gauss_params = self.__model[label]

            # Get P(Y=yi) where yi is the current label
            p_yi = gauss_params['label_prop']

            # Initialize P(x1|Y=yi) * P(x2|Y=yi) * ... * P(xn|Y=yi) * P(Y=yi)
            p = p_yi
            
            # For each feature column
            # This loop finds P(x|Y=yi) for each data point x in vector for the current label yi
            for gauss_param in gauss_params:

                if (gauss_param != 'label_prop'):

                    # At any step in this part of the for loop
                    #   gauss_param is the name of the current feature column we are on
                    #   gauss_params[gauss_param] is a tuple (mu_MLE, std_MLE) for each feature column
                    
                    # Get xi
                    xi = vector[gauss_param]

                    # Get MLE estimates from model
                    mu = gauss_params[gauss_param][0]
                    std = gauss_params[gauss_param][1]

                    # Calculate P(X=xi|Y=yi) and multiply it to our GNB likelihood
                    p *= self._gauss(xi,mu,std)
            
            label_likelihoods.update({label:p})

        # Return the label with the highest GNB likelihood
        return max(label_likelihoods, key=label_likelihoods.get)

    # Call fit(X,y) first
    # Given k data point in R^n_features, and the model being trained on some data, return k label predictions
    # data is array-like with n_features columns and k rows and MUST have the same column titles as the training data
    def predict(self,data):

        # Convert to pandas DataFrame and reset column names
        data = pd.DataFrame(data)
        data = self._reset_names(data)

        # Make sure the data has the right number of features
        if (data.shape[1] != self.n_features):
            raise Exception("Input data should have " + str(self.n_features) + " feature columns.")
        # Make sure the model was trained on something
        elif (len(self.__model) == 0):
            raise Exception("Please call fit(X,y) on some training data first.")
        # Make sure the model has the right number of features
        elif ((len(self.__model[0]) - 1) != self.n_features):
            raise Exception("Make sure you called fit(X,y) on the right data!")
        
        # Transpose the data so we can loop over columns instead of rows
        data = data.transpose()

        # Initialize our output
        out = np.array([])

        # For each data point in R^n_features
        for col in data:
            
            # Access it as a pandas Series
            vector = data[col]

            # Run the helper prediction function for this data point
            # Add the prediction to our output
            out = np.append(out, self.__predict_sub(vector))

        return out