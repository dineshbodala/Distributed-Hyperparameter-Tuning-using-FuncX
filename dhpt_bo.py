
from hyperopt import hp
from globus_compute_sdk import Executor

endpoint_id1 = ''
endpoint_id2= ''

def bayesian_opt1(space):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score 
    from sklearn.metrics import mean_squared_error, r2_score
    from hyperopt import fmin, tpe, STATUS_OK, Trials
    X = pd.read_csv('/home/edg1/scaled_features.csv')
    y = pd.read_csv('/home/edg1/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(space):
        model = RandomForestRegressor(
            max_depth=space['max_depth'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split'],
            n_estimators=space['n_estimators']
        )

        accuracy = cross_val_score(model, X_train, y_train, cv=3).mean()

        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    result = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=90, trials=trials)

    feat = {0: 'sqrt', 1: 'log2', 2: None}
    est = {0: 100, 1: 300, 2: 500, 3: 800, 4: 1000, 5: 1200}

    randomforest = RandomForestRegressor(
        max_depth=result['max_depth'],
        max_features=feat[result['max_features']],
        min_samples_leaf=result['min_samples_leaf'],
        min_samples_split=result['min_samples_split'],
        n_estimators=est[result['n_estimators']]
    ).fit(X_train, y_train)

    y_pred = randomforest.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)

    return y_pred,rmse,r_squared

def bayesian_opt2(space):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score 
    from sklearn.metrics import mean_squared_error, r2_score
    from hyperopt import fmin, tpe, STATUS_OK, Trials
    X = pd.read_csv('/home/edg4/scaled_features.csv')
    y = pd.read_csv('/home/edg4/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(space):
        model = RandomForestRegressor(
            max_depth=space['max_depth'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split'],
            n_estimators=space['n_estimators']
        )

        accuracy = cross_val_score(model, X_train, y_train, cv=3).mean()

        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    result = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=90, trials=trials)

    feat = {0: 'sqrt', 1: 'log2', 2: None}
    est = {0: 100, 1: 300, 2: 500, 3: 800, 4: 1000, 5: 1200}

    randomforest = RandomForestRegressor(
        max_depth=result['max_depth'],
        max_features=feat[result['max_features']],
        min_samples_leaf=result['min_samples_leaf'],
        min_samples_split=result['min_samples_split'],
        n_estimators=est[result['n_estimators']]
    ).fit(X_train, y_train)

    y_pred = randomforest.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)

    return y_pred,rmse,r_squared

space1 = {
    'max_depth': hp.choice('max_depth', list(range(1, 61))),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    'min_samples_split': hp.uniform('min_samples_split', 0, 1),
    'n_estimators': hp.choice('n_estimators', [100, 300, 500, 800, 1000, 1200])
}

space2 = {
    'max_depth': hp.choice('max_depth', list(range(1, 121))),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    'min_samples_split': hp.uniform('min_samples_split', 0, 1),
    'n_estimators': hp.choice('n_estimators', [50, 150, 250, 350, 450, 550])
}
with Executor(endpoint_id=endpoint_id1) as gce1, Executor(endpoint_id=endpoint_id2) as gce2:

    future1 = gce1.submit(bayesian_opt1,space1)
    future2 = gce2.submit(bayesian_opt2,space2)

    result1 = future1.result()
    result2 = future2.result()

    print(result1)
    print(result2)
