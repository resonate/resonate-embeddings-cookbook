from optuna.integration import OptunaSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score
import structlog
import os
from sklearn_cookbook.hyperparameter_space import HyperparameterSpace
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


logger = structlog.get_logger()

def preprocessing_data(input_path: str, embeddings_path: str):
    """
    Preprocessing for training data
    :param input_path: location of the input label matrix
    :param embeddings_path: location of the embeddings
    :return training inputs:
    """
    logger.info('Input paths:', path=input_path, path2=embeddings_path)
    input_path = os.path.join(input_path, evkey)
    logger.info('loading evkey data from path', path=input_path)
    raw_data = pd.read_parquet(input_path)
    # Embeddings act as a representation of input data in a smaller dimension space
    logger.info('loading embeddings data from path', path=embeddings_path)
    embeddings = pd.read_parquet(embeddings_path)
    # Join raw data with embeddings to get inputs and outputs to model
    training_data = pd.merge(embeddings, raw_data, on='rid', how='inner')
    logger.info('preparing data from training')
    x = pd.DataFrame(training_data['bottleneck'].tolist(), index=training_data.index)
    y = training_data['label']
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)
    return x, x_train, x_val, x_test, y_train, y_val, y_test, training_data


def select_k_best_features(x_train, y_train, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    x_train_selected = selector.fit_transform(x_train, y_train)
    return x_train_selected, selector


# Step 5: Reporting
def report_results(results):
    logger.info("Classifier Metrics:")
    for clf, metrics in results.items():
        logger.info(f"{clf}: {metrics}")
    best_classifier = max(results, key=lambda x: results[x]['AUC-ROC'])
    logger.info("Best Classifier based on ROC AUC", best_classifier=best_classifier)
    # logger.info("Metrics for classifier", best_classifier=results[results[]==best_classifier])


# Example usage
def main(input_path: str, embeddings_path: str, output_path: str, evkey: str, feature_selection: str):
    # Step 1: Data Preprocessing
    x, x_train, x_val, x_test, y_train, y_val, y_test, training_data = preprocessing_evkey_data(input_path,
                                                                                                embeddings_path, evkey)

    # Step 2: Feature Selection
    if feature_selection:
        x_train, selector = select_k_best_features(x_train, y_train)
        x_test = selector.transform(x_test)

    # Step 3: Hyperparameter Optimization
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        'KNeighbors': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

    results = {}

    # Iterate over classifiers
    for clf_name, clf in classifiers.items():
        logger.info('Current classifier being tested', classifier=clf_name, clf=clf)
        # Get hyperparameter search space from HyperparameterSpace class
        param_space = HyperparameterSpace.get_param_space(clf_name)
        # Perform hyperparameter tuning using Optuna
        optuna_search = OptunaSearchCV(clf, param_space, cv=3)
        optuna_search.fit(x_train, y_train)
        # Evaluate the model
        y_pred = optuna_search.predict(x_test)
        y_prob = optuna_search.predict(x_test)
        print(f"Classifier: {clf_name}")
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Compute AUC-ROC
        auc_roc = roc_auc_score(y_test, y_prob)
        report_dict['AUC-ROC'] = auc_roc
        results[clf_name] = report_dict
        # raw distribution, ROC, average precision
        # num_iterations 100-500 and use early stopping
        # shapley values

        report_dict_df = pd.DataFrame.from_dict(results)

        # Save results (implement saving model and metrics as needed)
        classifier_output_path = os.path.join(output_path, clf_name)
        os.makedirs(classifier_output_path, exist_ok=True)
        logger.info(f"Saving classifier output as csv", path=classifier_output_path)
        report_dict_df.to_csv(os.path.join(classifier_output_path, 'classification_report.csv'), index=False)
    report_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-path', type=str, help='Path to train, test files (path to train/..)')
    parser.add_argument('--output-path', type=str, help='Path to save trained model')
    parser.add_argument('--evkey', type=str, help='evkey')
    parser.add_argument('--embeddings-path', type=str, help='Path to embeddings')
    parser.add_argument('--feature-selection', type=bool, help='Boolean for feature selection')
    args = parser.parse_args()

    main(input_path=args.input_path,
         embeddings_path=args.embeddings_path,
         evkey=args.evkey,
         feature_selection=args.feature_selection,
         output_path=args.output_path)
