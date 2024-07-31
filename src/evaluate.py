from random_forest import RandomForest
from dataloader import DataLoader
import os
import yaml
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate(config_path):
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
    dataloader = DataLoader(config_path)
    dataloader.load_data()
    X_train,X_test,y_train,y_test = dataloader.load_preprocessed()
    dataloader.save_transformer()

    randomforest = RandomForest(config_path, dataloader, random_state=73)
    randomforest.load_model()

    models = {
        'rf':randomforest
         
        }


    for key,value in models.items():
        predictions = value.predict(X_test)
        print(confusion_matrix(predictions,y_test))
        print(accuracy_score(predictions,y_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="config file path")
    args = parser.parse_args()
    evaluate(args.config)