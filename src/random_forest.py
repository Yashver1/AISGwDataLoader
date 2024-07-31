from basemodel import BaseModel
from dataloader import DataLoader
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import yaml

class RandomForest(BaseModel):
    def __init__(self,config_path, dataloader, random_state=None):
        super().__init__(config_path,dataloader)
        self.model = RandomForestClassifier(random_state=random_state)
        self.alias = 'rf'
    
    def train(self,X,y):
        param_grid = self.config['models'][self.alias]['param_grid']
        random_grid = RandomizedSearchCV(self.model, param_distributions=param_grid, n_iter=self.config['experiment']['n_iter'], verbose=2, refit=True, cv=5, random_state=self.config['experiment']['random_state'], error_score='raise',n_jobs=10)
        random_grid.fit(X,y)
        self.model = random_grid.best_estimator_
        best_estimators = random_grid.best_params_
        return best_estimators


        