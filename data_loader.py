import os
import numpy as np
import pandas as pd


class DataLoader():

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def pickle_exists(self, name):
        '''
        Check is a pickle file exists in input directoy
        Args:
            name: a string denoting the name of a pickle file
        Output:
            A boolean
        '''
        return os.path.exists(f'{self.input_dir}{name}.pkl')

    def load_pickle(self, name):
        if self.pickle_exists(name):
            return pd.read_pickle(f'{self.input_dir}{name}.pkl')

    def save_pickle(self, df, name):
        df.to_pickle(f'{self.input_dir}{name}.pkl')

    def load(self, name):
        '''
        Checks if a pickle file already exists and loads it.
        Else, loads data from the source file.

        Args:
            name: a string denoting the name of a data source file
        Output:
            A pandas data frame
        '''
        # If pickle file exist, load it
        if self.pickle_exists(name):
            print(f'{name} exists')
            df = self.load_pickle(name)
        # Else, read from data source and create a pickle file
        else:
            df = pd.read_csv(f'{self.input_dir}{name}.csv', encoding="ISO-8859-1")
            self.save_pickle(df, name)

        return df

    def load_data(self):
        train = self.load('train')
        desc = self.load('product_descriptions')
        attr = self.load('attributes')
        return train, desc, attr

    def load_merged(self, name='merged'):
        '''
        Loads a data frame with where the three data sets are merged
        Output:
            A pandas dataframe
        '''
        if self.pickle_exists(name):
            df = self.load_pickle(name)
        else:
            train, desc, attr = self.load_data()

            # Keep only Brand Name from attributes
            brand = attr[attr.name == "MFG Brand Name"][[
                "product_uid", "value"]].rename(columns={"value": "brand"})

            # Merge the 3 dataframes into one
            df = pd.merge(train, desc, how='left', on='product_uid')
            df = pd.merge(df, brand, how='left', on='product_uid')

            self.save_pickle(df, name)

        return df

    def directory(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
