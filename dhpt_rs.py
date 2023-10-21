import itertools
import numpy as np
from globus_compute_sdk import Executor

endpoint_id1 = 'cc86c44c-12a8-4039-918a-bc64d4ba8599'
endpoint_id2= 'd241b648-cbae-4562-b978-a438af732024'

def random_search1(subspaces, num_samples):
    import random
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    sampled_subspaces = random.sample(subspaces, num_samples)
    best_rmse = float('inf')
    best_r_squared = 0
    best_params = None

    X = pd.read_csv('/home/edg1/scaled_features.csv')
    y = pd.read_csv('/home/edg1/y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for subspace in sampled_subspaces:
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

def random_search2(subspaces, num_samples):
    import random
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    sampled_subspaces = random.sample(subspaces, num_samples)
    best_rmse = float('inf')
    best_r_squared = 0
    best_params = None

    X = pd.read_csv('/home/edg4/scaled_features.csv')
    y = pd.read_csv('/home/edg4/y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for subspace in sampled_subspaces:
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

# Generate all possible combinations of hyperparameters
rf_params = {
    'max_depth': ['None'] + list(np.arange(10, 110, 10)),
    'max_features': [ 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [100, 200, 300, 400, 500]
}

param_combinations = list(itertools.product(*rf_params.values()))

# Convert the combinations into a list of dictionaries, handling 'None'
subspaces = [{'max_depth': str(p[0]),
              'max_features': p[1],
              'min_samples_split': p[2],
              'min_samples_leaf': p[3],
              'n_estimators': p[4]} for p in param_combinations]

total_combinations = len(subspaces)
subspace1 = subspaces[:total_combinations // 2]
subspace2 = subspaces[total_combinations // 2:]
num_samples = 100  


with Executor(endpoint_id=endpoint_id1) as gce1, Executor(endpoint_id=endpoint_id2) as gce2:

    future1 = gce1.submit(random_search1,subspace1,num_samples)
    future2 = gce2.submit(random_search2,subspace2,num_samples)

    result1 = future1.result()
    result2 = future2.result()

    print(result1)
    print(result2)