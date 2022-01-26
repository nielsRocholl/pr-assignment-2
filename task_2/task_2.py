import os
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import fbeta_score

#import logistic regression from sklearn
from sklearn.linear_model import LogisticRegression

def represent_classes_equally(df: pd.DataFrame) -> pd.DataFrame:
    # Make training set represent each class equally
    grouped_df = df.groupby('Class')
    fraud = grouped_df.get_group(1)
    non_fraud = grouped_df.get_group(0).sample(frac=1).head(len(fraud))
    data = pd.concat([non_fraud, fraud]).sample(frac=1)
    return data


def train_test_set() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TRAIN_SIZE = 0.8
    LABELED_SIZE = 0.3

    # Load data and drop irrelevant columns
    df = pd.read_csv("creditcard.csv")
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Split the data into train and test
    grouped_df = df.groupby('Class')
    train_0, test_0 = train_test_split(grouped_df.get_group(0), train_size=TRAIN_SIZE, shuffle=True)
    train_1, test_1 = train_test_split(grouped_df.get_group(1), train_size=TRAIN_SIZE, shuffle=True)

    # Merge the different classes
    X_train = pd.concat([train_0, train_1])
    X_test = pd.concat([test_0[:len(test_1)], test_1])

    # Split the train data into labeled and unlabeled
    X_train_labeled, X_train_unlabeled = train_test_split(X_train, train_size=LABELED_SIZE, shuffle=True)

    # Balance classes in labelled sets and delete class from unlabeled set
    X_train_labeled = represent_classes_equally(X_train_labeled)
    X_train_unlabeled.assign(Class=-1)

    # X_train_unlabeled.assign(Class=-1)
    return X_train_labeled, X_train_unlabeled, X_test


def baseline_method(train: pd.DataFrame, test: pd.DataFrame) -> None:
    # Train the model
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(train.drop(["Class"], axis=1), train['Class'])

    # predict the test set class labels and return F2 Score
    pred_y = model.predict(test.drop(['Class'], axis=1))
    return fbeta_score(test["Class"], pred_y, beta=1)


def semi_supervised_method(train_labeled: pd.DataFrame, train_unlabeled: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Use KNN for label propagation
    label_prop = LabelPropagation(kernel="knn", n_neighbors=20, n_jobs=-1).fit(train_labeled, train_labeled['Class'])

    # Predict the labels of the unlabeled data and add them to the labeled data
    train_unlabeled.assign(Class=label_prop.predict(train_unlabeled))
    train_unlabeled = represent_classes_equally(train_unlabeled)

    # Return the now fully labeled dataset
    return pd.concat([train_labeled, train_unlabeled])


def main():
    # Load the data
    X_train_labeled, X_train_unlabeled, X_test = train_test_set()

    # 2 Run the baseline method on only the labeled dataset
    f2_base = baseline_method(X_train_labeled, X_test)

    # 3 Assign labels to unlabeled set using the semi-supervised method
    set_with_assigned_labels = semi_supervised_method(X_train_labeled, X_train_unlabeled, X_test)

    # 4 Run the baseline method on the now fully labeled training dataset
    f2_semi_supervised = baseline_method(set_with_assigned_labels, X_test)

    # Return the F2 scores of the baseline method and the semi-supervised method
    return f2_base, f2_semi_supervised


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    f2 = {'base': [], 'semi_supervised': []}
    for idx in range(100):
        f2_base, f2_semi_supervised = main()
        f2['base'].append(f2_base)
        f2['semi_supervised'].append(f2_semi_supervised)
        print(f'\rBase: {np.mean(f2["base"]): .3f}\tSemi: {np.mean(f2["semi_supervised"]): .3f}', end='')
    
    print(f'\rF2 baseline: {np.mean(f2["base"])}, {np.std(f2["base"])}')
    print(f'F2 semi-supervised: {np.mean(f2["semi_supervised"])}, {np.std(f2["semi_supervised"])}')
