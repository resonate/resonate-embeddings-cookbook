from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution


class HyperparameterSpace:
    @staticmethod
    def get_param_space(classifier_name):
        if classifier_name == 'RandomForest':
            return {
                'n_estimators': IntDistribution(10, 100),
                'max_depth': IntDistribution(1, 50),
                'min_samples_split': IntDistribution(2, 20),
                'min_samples_leaf': IntDistribution(1, 10),
                'max_features': CategoricalDistribution(['auto', 'sqrt', 'log2'])
            }
        elif classifier_name == 'LogisticRegression':
            return {
                'C': FloatDistribution(1e-3, 1e3),
                'solver': CategoricalDistribution(['liblinear', 'saga']),
                'penalty': CategoricalDistribution(['l1', 'l2'])
            }
        elif classifier_name == 'KNeighbors':
            return {
                'n_neighbors': IntDistribution(1, 50),
                'weights': CategoricalDistribution(['uniform', 'distance']),
                'algorithm': CategoricalDistribution(['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': IntDistribution(10, 50),
                'p': IntDistribution(1, 2)
            }
        elif classifier_name == 'GradientBoosting':
            return {
                'n_estimators': IntDistribution(10, 200),
                'learning_rate': FloatDistribution(0.01, 0.1),
                'max_depth': IntDistribution(1, 10),
                'min_samples_split': IntDistribution(2, 20),
                'min_samples_leaf': IntDistribution(1, 10),
                'subsample': FloatDistribution(0.5, 1.0),
                'max_features': CategoricalDistribution(['auto', 'sqrt', 'log2'])
            }
        elif classifier_name == 'XGBoost':
            return {
                'n_estimators': IntDistribution(10, 200),
                'learning_rate': FloatDistribution(0.01, 0.1),
                'max_depth': IntDistribution(1, 10),
                'subsample': FloatDistribution(0.5, 1.0),
                'colsample_bytree': FloatDistribution(0.5, 1.0),
                'reg_alpha': FloatDistribution(1e-3, 10.0),
                'reg_lambda': FloatDistribution(1e-3, 10.0)
            }
        else:
            raise ValueError("Classifier not supported")
