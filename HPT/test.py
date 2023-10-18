import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import optuna

def objective(trial):
    # Load data
    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_features': trial.suggest_int('max_features', 1, 13),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 11),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 11),
        'criterion': trial.suggest_categorical('criterion', ['mse', 'mae'])
    }
    if rf_params['criterion'] == 'mse':
        rf_params['criterion'] = 'squared_error'
    elif rf_params['criterion'] == 'mae':
        rf_params['criterion'] = 'absolute_error'
    rf = RandomForestRegressor(**rf_params, random_state=0)
    rf.fit(X_train, y_train.values.ravel())
    predictions = rf.predict(X_test)
    r_squared = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return r_squared, rmse

study = optuna.create_study(direction='maximize')
trial = study.ask()

# Call the objective function with the trial object
result = objective(trial)

# Print or use the result returned by the function
print(result)