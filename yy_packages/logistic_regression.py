# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from vectorize import text_preparation

class LogisticModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic_regression(self, X, x, Y, y):
        model = LogisticRegression(max_iter=500, solver="liblinear")
        model.fit(X, Y)

        coefficients = model.coef_[0]
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report, coefficients

    def train_and_test(self):
        accuracy_list = []
        report_list = []
        features = pd.DataFrame()

        for model_type in ['default', 'stemming', 'lemmatizing']:
            vetorized_ob = text_preparation(self.X_train, self.X_test)
            X_train_new, X_test_new, text_features = vetorized_ob.vectorizing(model_type)
            accuracy, report, coefficients = self.logistic_regression(X_train_new, X_test_new, self.y_train, self.y_test)

            accuracy_list.append(accuracy)
            report_list.append(report)
            feature_df = pd.DataFrame(
                {f'feature_{model_type}': text_features, f'{model_type}_coefficient': coefficients})
            feature_20 = feature_df.sort_values(f'{model_type}_coefficient', ascending=False).head(20).reset_index(
                drop=True)
            features = pd.concat([features, feature_20], axis=1)
        return accuracy_list, report_list, features





