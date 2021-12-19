import numpy as np
import pandas as pd
import lifelines
from sklearn.model_selection import train_test_split


def load_data_from_csv(threshold, test_size=0.2):
    """
    Loads data from csv file and returns data is available for the threshold
    threshold: int (number of years)
    """
    X = pd.read_csv('data/NHANESI_subset_X.csv', index_col=0)
    y = np.array(pd.read_csv('data/NHANESI_subset_y.csv')['y'])

    df = X.copy()
    # Create a column for the year
    df.loc[:, 'time'] = y
    # Check the people who death or not
    df.loc[:, 'death'] = np.ones(len(X))
    df.loc[df.time < 0, 'death'] = 0
    df.loc[:, 'time'] = np.abs(df.time)
    df = df.dropna(axis='rows')
    mask = (df.time > threshold) | (df.death == 1)
    df = df[mask]
    X = df.drop(['time', 'death'], axis='columns')
    y = df.time < threshold

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def c_index(y_true, scores):
    return lifelines.utils.concordance_index(y_true, scores)
