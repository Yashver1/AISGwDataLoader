from dataloader import DataLoader
from random_forest import RandomForest
from sklearn.metrics import accuracy_score

config_path = './src/config.yaml'
dl = DataLoader(config_path)
dl.load_data()
X_train, X_test, y_train, y_test = dl.load_preprocessed()
dl.save_transformer()
dl.load_transformer()

randomforest = RandomForest(config_path, dl, random_state=73)
best_params = randomforest.train(X_train, y_train)
predictions = randomforest.predict(X_test)

print(best_params)
print(accuracy_score(y_test, predictions))
