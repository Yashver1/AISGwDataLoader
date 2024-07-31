import yaml
import joblib
import os

class BaseModel():
    def __init__(self,config_path,dataloader):
        with open(config_path,"r") as file:
            self.config = yaml.safe_load(file)
        self.model = None
        self.alias = None
        self.dataloader = dataloader

    def train(self):
        raise NotImplementedError

    def predict(self, X, transform=None):
        if transform is not None:
            X = self.dataloader.data_augmentation(X)
        predictions = self.model.predict(X)
        return predictions

    def save_model(self):
        current_path = os.path.dirname(__file__)
        relative_path = os.path.join(current_path, self.config['models'][self.alias]['model_path'])
        abs_path = os.path.abspath(relative_path)
        joblib.dump(self.model,abs_path)

    def load_model(self):
        current_path = os.path.dirname(__file__)
        relative_path = os.path.join(current_path, self.config['models'][self.alias]['model_path'])
        abs_path = os.path.abspath(relative_path)
        self.model = joblib.load(abs_path)

