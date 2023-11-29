import itertools
import numpy as np
from globus_compute_sdk import Executor

endpoint_id1 = ''
endpoint_id2= ''

def grid_search1(subspaces):
    import random
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    best_rmse = float('inf')
    best_r_squared = 0
    best_params = None

    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for subspace in subspaces:
        rf = RandomForestRegressor()

        # Set max_depth to None if 'None', or a very large number otherwise
        subspace['max_depth'] = None if subspace['max_depth'] == 'None' else int(subspace['max_depth'])
        rf.set_params(**subspace)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r_squared = rf.score(X_test, y_test)

        if rmse < best_rmse:
            best_rmse = rmse
            best_r_squared = r_squared
            best_params = subspace

    return best_params, best_rmse, best_r_squared

def grid_search2(subspaces):
    import random
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    best_rmse = float('inf')
    best_r_squared = 0
    best_params = None

    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for subspace in subspaces:
        rf = RandomForestRegressor()

        # Set max_depth to None if 'None', or a very large number otherwise
        subspace['max_depth'] = None if subspace['max_depth'] == 'None' else int(subspace['max_depth'])
        rf.set_params(**subspace)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r_squared = rf.score(X_test, y_test)

        if rmse < best_rmse:
            best_rmse = rmse
            best_r_squared = r_squared
            best_params = subspace

    return best_params, best_rmse, best_r_squared

rf_params={
    'max_depth': [2,4,6,8,10],
    'max_features': ['sqrt','log2'],
    'min_samples_leaf': [4, 6, 8, 12],
    'min_samples_split': [5, 7, 10, 14],
    'n_estimators': [i for i in range(100,2000,200)]}

param_combinations = list(itertools.product(*rf_params.values()))

subspaces = [{'max_depth': str(p[0]),
              'max_features': p[1],
              'min_samples_split': p[2],
              'min_samples_leaf': p[3],
              'n_estimators': p[4]} for p in param_combinations]

total_combinations = len(subspaces)
subspace1 = subspaces[:total_combinations // 2]
subspace2 = subspaces[total_combinations // 2:]

print(grid_search1(subspace1))
print(grid_search2(subspace2))
