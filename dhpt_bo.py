import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

# Load your data
X = pd.read_csv('scaled_features.csv')
y = pd.read_csv('y.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define multiple search spaces for different parts of hyperparameters
pbounds1 = {
    'max_depth': (10, 50),
    'max_features': (0.1, 0.9),
    'min_samples_split': (2, 5),
    'min_samples_leaf': (1, 2),
    'n_estimators': (100, 300)
}

pbounds2 = {
    'max_depth': (50, 110),
    'max_features': (0.1, 0.9),
    'min_samples_split': (5, 10),
    'min_samples_leaf': (2, 4),
    'n_estimators': (300, 500)
}

# Define the objective function for Bayesian optimization
def objective_function(max_depth, max_features, min_samples_split, min_samples_leaf, n_estimators):
    rf = RandomForestRegressor(
        max_depth=int(max_depth),
        max_features=max_features,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        n_estimators=int(n_estimators)
    )

    kfold = KFold(n_splits=5)
    rmse_sum = 0
    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_sum += rmse

    return -rmse_sum / kfold.n_splits  # Note the negative sign for minimization

# Create separate Bayesian optimization instances for different search spaces
bo1 = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds1,
    random_state=42
)

bo2 = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds2,
    random_state=42
)

# Set Gaussian process parameters using set_gp_params method
bo1.set_gp_params(normalize_y=True)
bo2.set_gp_params(normalize_y=True)

# Create UtilityFunction with kappa=2.576 for UCB acquisition function
utility1 = UtilityFunction(kind="ucb", kappa=2.576, xi=0.0)
utility2 = UtilityFunction(kind="ucb", kappa=2.576, xi=0.0)

# Run Bayesian optimization for each search space
bo1.maximize(init_points=5, n_iter=100, acq="ucb", utility_function=utility1)
bo2.maximize(init_points=5, n_iter=100, acq="ucb", utility_function=utility2)

# Get the best hyperparameters from each Bayesian optimization run
best_params1 = bo1.max['params']
best_params2 = bo2.max['params']

# Train Random Forest models using the best hyperparameters from each run
rf1 = RandomForestRegressor(**best_params1)
rf2 = RandomForestRegressor(**best_params2)

# Train the models using the training data
rf1.fit(X_train, y_train.values.ravel())
rf2.fit(X_train, y_train.values.ravel())

# Evaluate the performance of each model on the test set
y_pred1 = rf1.predict(X_test)
y_pred2 = rf2.predict(X_test)

# Calculate the RMSE and R2 score of each model
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred1))
r2_1 = r2_score(y_test, y_pred1)

rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
r2_2 = r2_score(y_test, y_pred2)

# Print the RMSE and R2 score of each model
print('RMSE of model 1:', rmse1)
print('R2 score of model 1:', r2_1)

print('RMSE of model 2:', rmse2)
print('R2 score of model 2:', r2_2)
