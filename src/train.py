from random_forest import RandomForest
from dataloader import DataLoader
import os
import yaml
import argparse

def train(config_path):
    config = None
    with open(config_path,"r") as file:
        config = yaml.safe_load(config_path)

    dataloader = DataLoader(config_path)
    dataloader.load_data()
    X_train,X_test,y_train,y_test = dataloader.load_preprocessed()
    dataloader.save_transformer()
    randomforest = RandomForest(config_path, dataloader, random_state=73)

    models = {
        'rf':randomforest
         
        }


    for key,value in models.items():
        print(f"Training {value}")
        best_estimator = value.train(X_train,y_train)
        print(best_estimator)
        value.save_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Train", description="Train selected Models")
    parser.add_argument('--config', type=str, required=True, help="config file path")
    args = parser.parse_args()
    train(args.config)
    
