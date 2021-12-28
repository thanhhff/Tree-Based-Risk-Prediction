from utils import load_data_from_csv


if __name__ == "__main__":
    # Load data from csv, threshold is 10 years
    X_train, y_train, X_test, y_test = load_data_from_csv(threshold=10)

    print(X_train)
