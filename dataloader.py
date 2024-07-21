import yaml
import os 
import sqlite3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,Normalizer
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader():
    def __init__(self,config_path):
        with open(config_path,"r") as file:
            self.config = yaml.safe_load(file)
        self.transformer = None

    def load_data(self):
        db_path = self.config['data']['db_path'] 
        script_path = os.path.dirname(__file__) #file is current script, will get directory name
        relative_db_path = os.path.join(script_path,db_path) # appends current directory with db path
        abs_db_path = os.path.abspath(relative_db_path) # abspath from root
        conn = sqlite3.connect(abs_db_path)
        query = 'SELECT * FROM calls'
        data = pd.read_sql_quey(query,conn)
        conn.close()
        return data
    
    def augment_data(self,X,y=None,fit=False): #y=None for actual prediction
        X['Call Type'] == X['Call Type'].apply(lambda x: 'Whatsapp' if x=='Whatsapp' else x)
        X['Financial Loss'] = X['Financial Loss'].fillna(self.config['impute_na']['financial_loss'])

        FL_filter = X['Financial Loss']>=0 
        X = X[FL_filter]
        if y is not None:
            y = y[FL_filter]  # X['Financial Loss']>=0 simply returns index hence u can filter y but must store in temp first so u can keep the state
        
        median_call_duration = X.loc[X['Call Duration']>=0,'Call Duration'].median() #loc is loc['index of rows','column names']
        X.loc[X['Call Duration'< 0,'Call Duration']] = median_call_duration

        if fit:
            self.transformer = ColumnTransformer(
                transformers=[
                    ('cat',OneHotEncoder(),self.config['data']['categorical'])
                    ('num',Normalizer(),self.config['data']['numerical'])
                ]
            )
            X = self.transformer.fit_transform(X)
        else:
            if self.transformer is None:
                raise ValueError("Transformer not fit")
            else:
                X = self.transformer.transform(X)

        X['Call Duration'] = np.sqrt(X['Call Duration'])
        X['Call Frequency'] = np.sqrt(X['Call Frequency'])
        X['Previous Contact Count'] = np.sqrt(X['Previous Contact Count'])

        if y:
            return X,y
        else:
            return X
        

    def augment_label(self,y):
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel() #Need .ravel() as it is each element is enclosed in a list despite being one of e.g. [1]
        return y

    def preprocess(self,data): # will settle train_test_split and use augment data
        X = data.drop(columns=self.config['data']['label'])
        y = data[self.config['data']['label']]
        X_train,X_test,y_train,y_test = train_test_split(X,y,data,test_size=0.2,stratify=data['Scam Call'],random_state=73)
        X_train,y_train = self.augment_data(X_train,y_train,fit=True)
        X_test,y_test = self.augment_data(X_test,y_test,fit=False)
        y_train = self.augment_label(y_train)
        y_test = self.augment_label(y_test)

        return X_train,X_test,y_train,y_test



        

        





    

        