import os
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_set() -> tuple:
    TRAIN_SIZE = 0.8
    LABELED_SIZE = 0.3

    df = pd.read_csv("creditcard.csv")
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    grouped_df = df.groupby('Class')

    # Split the data into train and test
    train_0, test_0 = train_test_split(grouped_df.get_group(0), train_size=TRAIN_SIZE, shuffle=True)
    train_1, test_1 = train_test_split(grouped_df.get_group(1), train_size=TRAIN_SIZE, shuffle=True)
    
    # Merge the different classes
    X_train = pd.concat([train_0, train_1])
    X_test = pd.concat([train_0, train_1])

    # Split the train data into labeled and unlabeled
    X_train_labeled, X_train_unlabeled = train_test_split(X_train, train_size=LABELED_SIZE, shuffle=True)
    
    return X_train_labeled, X_train_unlabeled, X_test


def baseline_method(train: pd.DataFrame, test: pd.DataFrame) -> None:
    pass


def semi_supervised_method(train_labeled: pd.DataFrame, train_unlabeled: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pass


def main():
    X_train_labeled, X_train_unlabeled, X_test = train_test_set()

    # 2
    baseline_method(X_train_labeled, X_test)

    # 3
    X_with_predicted_labels = semi_supervised_method(X_train_labeled, X_train_unlabeled, X_test)

    # 4
    baseline_method(pd.concat([X_train_labeled, X_with_predicted_labels]), X_test)


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()