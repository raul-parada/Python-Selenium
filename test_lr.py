import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, roc_auc_score
import pandas as pd
import random
import numpy as np

# load the data into a DataFrame
@pytest.fixture(scope="module")
def dataset():
    # load the dataset
    df = pd.read_csv('5G_Sliced.csv')
    # separate the features (X) and the label (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def test_missing_data(dataset):
    # check for missing data in the dataset
    X, y = dataset
    assert not X.isnull().values.any(), "There are missing values in the dataset"
    assert not y.isnull().values.any(), "There are missing labels in the dataset"

def test_imbalanced_labels(dataset):
    # check for imbalanced labels in the dataset
    X, y = dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    value_counts = y.value_counts()
    majority_class_count = value_counts.max()
    minority_class_count = value_counts.min()
    assert minority_class_count >= 0.05 * majority_class_count, "The dataset is too imbalanced"

def generate_stochastic_data():
    column1 = random.choices(range(1, 23), weights=[0.047872]*20 + [0.021277]*2, k=1)[0]
    column2 = random.choices([1, 2], weights=[0.531915, 0.468085], k=1)[0]
    column3 = random.choices(range(1, 8), weights=[0.142857]*7, k=1)[0]
    column4 = random.choices(range(24), weights=[0.041667]*24, k=1)[0]
    column5 = random.choices([1, 2], weights=[0.558511, 0.441489], k=1)[0]
    column6 = random.choices([1, 2, 3], weights=[0.281915, 0.271277, 0.446809], k=1)[0]
    column7 = random.choices(range(1, 8), weights=[0.234043, 0.234043, 0.212766, 0.170213, 0.053191, 0.053191, 0.053191], k=1)[0]
    column8 = random.choices([1, 2, 3], weights=[0.531915, 0.234043, 0.234043], k=1)[0]

    sample_row = [column1, column2, column3, column4, column5, column6, column7, column8]
    X_test = np.array(sample_row[:-1]) # X_test is all columns except the last
    y_test = sample_row[-1] # y_test is the last column
    return X_test, y_test

def train_classification_model():
    # Test that the model can fit the data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)    
    assert model.score(X, y) > 0.7
    X_test, y_test = generate_stochastic_data()
    y_pred = model.predict(X_test)
    assert np.array_equal(y_pred, y_test), "y_pred is not equal to y_test"
    return model 

def evaluation():
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_classification_model()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.75, f"Accuracy is {accuracy}, expected 0.75 or higher"
    precision = precision_score(y_test, y_pred)
    assert precision >= 0.6666666666666666, f"Precision is {precision}, expected 0.6667 or higher"
    recall = recall_score(y_test, y_pred)
    assert recall == 1.0, f"Recall is {recall}, expected 1.0"
    f1 = f1_score(y_test, y_pred)
    assert f1 >= 0.8, f"F1 Score is {f1}, expected 0.8 or higher"
    mse = mean_squared_error(y_test, y_pred)
    assert round(mse, 2) <= 0.23, f"MSE is {mse}, expected 0.23 or lower"
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    assert round(rmse, 2) <= 0.48, f"RMSE is {rmse}, expected 0.48 or lower"
    auc = roc_auc_score(y_test, y_pred)
    assert round(auc, 2) >= 0.92, f"AUC is {auc}, expected 0.92 or higher"
    mae = mean_absolute_error(y_test, y_pred)
    assert round(mae, 2) <= 0.39, f"MAE is {mae}, expected 0.39 or lower"
    kappa = cohen_kappa_score(y_test, y_pred)
    assert round(kappa, 2) <= 0.67, f"Cohen's Kappa Score is {kappa}, expected 0.67 or lower"