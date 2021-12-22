import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import load_data_from_csv
from sklearn.model_selection import train_test_split

def get_training_data():
    X_train_org, X_test, y_train_org, y_test = load_data_from_csv(threshold=10, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size=0.25, random_state=42)

    return X_train, y_train, X_val, y_val
