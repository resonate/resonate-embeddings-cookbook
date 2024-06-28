from sklearn.base import BaseEstimator, TransformerMixin
from optuna.integration import OptunaSearchCV
from sklearn.metrics import classification_report


class OptunaSearchCVAndReport(BaseEstimator, TransformerMixin):
    def __init__(self, classifier, param_space, cv=3):
        self.optuna_search = None
        self.report = None
        self.classifier = classifier
        self.param_space = param_space
        self.cv = cv

    def fit(self, x, y):
        # Perform hyperparameter tuning using Optuna
        self.optuna_search = OptunaSearchCV(self.classifier, self.param_space, cv=self.cv)
        self.optuna_search.fit(x, y)
        return self

    def transform(self, x, y):
        # Evaluate the model and return classification report
        y_pred = self.optuna_search.predict(x)
        self.report = classification_report(y, y_pred, output_dict=True)
        return x

    def get_classification_report(self):
        return self.report
