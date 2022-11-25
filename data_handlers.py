import numpy as np
import pandas as pd


class Penguins:

    def __init__(self, path, preprocess=True):
        self.data = pd.read_csv(path)

        if preprocess:
            self.preprocess_data()

    def preprocess_data(self):
        # Replace Na values with the last valid value of their column
        self.data.fillna('pad', inplace=True)
        # Convert `male` to 1, `female` to 0
        self.data['gender'] = np.where(self.data['gender'] == 'male', 1, 0)

        # Convert `flipper_length_mm` to Cm
        self.data['flipper_length_mm'] = self.data['flipper_length_mm'] / 10
        # Convert `body_mass_g` to Kg
        self.data['body_mass_g'] = self.data['body_mass_g'] / 1000
        
        # Convert all values to fractions
        func = lambda x: x / 100
        self.data['flipper_length_mm'] = self.data['flipper_length_mm'].apply(func)
        self.data['bill_length_mm'] = self.data['bill_length_mm'].apply(func)
        self.data['bill_depth_mm'] = self.data['bill_depth_mm'].apply(func)
        self.data['body_mass_g'] = self.data['body_mass_g'].apply(func)

        # Convert species' names to 0-index classes
        labels = {"Adelie": 0, "Gentoo": 1, "Chinstrap": 2}
        self.data["species"] = self.data["species"].apply(lambda x: labels[x])

    def partition_data(self):
        # Create empty DataFrames
        training = pd.DataFrame()
        testing = pd.DataFrame()

        y_label = "species"

        # Group by the label, then add 30 training rows to the training DataFrame
        # and 20 test rows to the testing frame
        for _, group in self.data.groupby(y_label):
            training = pd.concat([training, group.iloc[:30]], ignore_index=True)
            testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

        # Randomly shuffle the data
        training = training.sample(frac=1)
        testing = testing.sample(frac=1)

        # Return the feature columns and label column separately for each DataFrame, such that the return values
        # are training_feature_columns, training_label_column, testing_feature_columns, testing_label_coloumn
        # Inspired by scikit-learn train_test_split function
        return (
            training.loc[:, training.columns != y_label],
            training[y_label],
            testing.loc[:, testing.columns != y_label],
            testing[y_label]
        )


class MNIST:

    def __init__(self, train_path, test_path, preprocess=True):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

        if preprocess:
            self.preprocess_data()

    def preprocess_data(self):
        filt = self.train_data.columns != "label"
        func = lambda x: x / 1000

        self.train_data.loc[:, filt] = self.train_data.loc[:, filt].applymap(func)
        self.test_data.loc[:, filt] = self.test_data.loc[:, filt].applymap(func)

    def partition_data(self):
        y_label = "label"
        filt = self.train_data.columns != y_label

        x_train = self.train_data.loc[:, filt]
        y_train = self.train_data[y_label]

        x_test = self.test_data.loc[:, filt]
        y_test = self.test_data[y_label]

        return x_train, y_train, x_test, y_test
