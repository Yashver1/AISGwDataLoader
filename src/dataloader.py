import yaml
import sqlite3
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib


class DataLoader():
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.data = None
        self.transformer = None

    def load_data(self):
        current_path = os.path.dirname(__file__)
        relative_path = os.path.join(
            current_path, self.config['data']['db_path'])
        abs_path = os.path.abspath(relative_path)
        con = sqlite3.connect(abs_path)
        self.data = pd.read_sql(sql='SELECT * FROM calls', con=con)

    # y is none for rl predict. fit is false for test phase
    def data_augmentation(self, X, y=None, fit=False):
        X['Call Type'] = X['Call Type'].apply(
            lambda x: 'Whatsapp' if x == 'Whatsapp' else x)
        X['Financial Loss'] = X['Financial Loss'].fillna(0)

        FL_idx = X['Financial Loss'] > 0
        X = X[FL_idx]
        if y is not None:
            y = y[FL_idx]

        X = X.drop(columns=self.config['experiment']['drop_features'])
        median_call_duration = X.loc[X['Call Duration']
                                     > 0, 'Call Duration'].median()
        X.loc[X['Call Duration'] < 0,
              'Call Duration'] = median_call_duration
        X[[
            'Call Duration',
            'Call Frequency',
            'Previous Contact Count'
        ]] = np.sqrt(
            X[['Call Duration', 'Call Frequency', 'Previous Contact Count']])

        if fit:
            transformer = ColumnTransformer(
                [("norm",
                  Normalizer(),
                  self.config['experiment']['numerical_features']),
                 ("one_hot",
                  OneHotEncoder(),
                  self.config['experiment']['categorical_features'])])
            X = transformer.fit_transform(X)
            self.transformer = transformer
        else:
            if self.transformer is None:
                raise ValueError("Transformer not set.")
            X = self.transformer.transform(X)
        if y is not None:
            lb = LabelEncoder()
            y = lb.fit_transform(y.values.ravel())

        if y is not None:
            return X, y
        else:
            return X

    def load_preprocessed(self):
        if self.data is None:
            raise ValueError("No Data Loaded.")

        X = self.data.drop(columns=self.config['experiment']['label'])
        y = self.data[self.config['experiment']['label']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=73)
        X_train, y_train = self.data_augmentation(
            X_train, y=y_train, fit=True)
        X_test, y_test = self.data_augmentation(
            X_test, y=y_test, fit=False)

        return X_train, X_test, y_train, y_test

    def save_transformer(self):
        current_path = os.path.dirname(__file__)
        relative_path = os.path.join(
            current_path, self.config['dataloader']['transformer_path'])
        abs_path = os.path.abspath(relative_path)
        joblib.dump(self.transformer, abs_path)

    def load_transformer(self):
        current_path = os.path.dirname(__file__)
        relative_path = os.path.join(
            current_path, self.config['dataloader']['transformer_path'])
        abs_path = os.path.abspath(relative_path)
        self.transformer = joblib.load(abs_path)
