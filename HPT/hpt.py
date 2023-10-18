from datetime import datetime

def bayesian_opt1():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, train_test_split,GridSearchCV,cross_val_score
    from sklearn.metrics import mean_squared_error
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

    X = pd.read_csv('/home/edg4/scaled_features.csv')
    y = pd.read_csv('/home/edg4/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
    space = {
        'max_depth':hp.choice('max_depth', list(range(1, 121))), 
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split': hp.uniform('min_samples_split', 0, 1),
        'n_estimators': hp.choice('n_estimators', [100, 300, 500, 800, 1000, 1200])
    }

    def objective(space):
        model = RandomForestRegressor(max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
        accuracy = cross_val_score(model, X_train, y_train, cv = 3).mean()

        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials = Trials()
    result = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 90,
            trials= trials)

    
    feat = {0: 'sqrt', 1: 'log2', 2: None }

    est = {0: 100, 1: 300, 2: 500, 3: 800, 4: 1000,5:1200}


    randomforest = RandomForestRegressor( 
                                       max_depth = result['max_depth'], 
                                       max_features = feat[result['max_features']], 
                                       min_samples_leaf = result['min_samples_leaf'], 
                                       min_samples_split = result['min_samples_split'], 
                                       n_estimators = est[result['n_estimators']]
                                      ).fit(X_train,y_train)
    

    y_pred=randomforest.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    y_pred=y_pred.reshape(y_test.shape)
    SS_Residual = sum((y_test.values-y_pred)**2)       
    SS_Total = sum((y_test.values-np.mean(y_test))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


    return y_pred, rmse, r_squared,SS_Residual,SS_Total,adjusted_r_squared





def random_search():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, train_test_split,GridSearchCV,cross_val_score
    from sklearn.metrics import mean_squared_error
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
    import pandas as pd
    import numpy as np
    X = pd.read_csv('/home/edg4/scaled_features.csv')
    y = pd.read_csv('/home/edg4/y.csv')    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    rf = RandomForestRegressor()


    rf_params = {
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100, 200, 300, 400, 500]
    }


    random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, 
                                   n_iter=100, cv=3, n_jobs=-1, random_state=42)

    random_search.fit(X_train, y_train)


    best_rf_model = random_search.best_estimator_


    y_pred = best_rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = best_rf_model.score(X_test, y_test)

    return y_pred, rmse, r_squared

def hyperband():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import optuna
    def objective(trial):
    # Load data
        X = pd.read_csv('/home/edg4/scaled_features.csv')
        y = pd.read_csv('/home/edg4/y.csv')

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
        r_squared = rf.score(X_test, y_test)
        return r_squared


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) 


    best_params = study.best_params
    best_r_squared = study.best_value

    return best_params, best_r_squared

def optimize_random_forest():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

    # Load the input data
    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the function to optimize and evaluate the model
    #optimize_random_forest(X_train, y_train, X_test, y_test)

    # Define the hyperparameter search space
    space = {
        'max_depth': hp.choice('max_depth', list(range(1, 121))),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split': hp.uniform('min_samples_split', 0, 1),
        'n_estimators': hp.choice('n_estimators', [100, 300, 500, 800, 1000, 1200])
    }

    # Use TPE optimization


    def tpe_optimization(X_train, y_train, space, max_evals=90):
        def objective(space):
            model = RandomForestRegressor(max_depth=space['max_depth'],
                                         max_features=space['max_features'],
                                         min_samples_leaf=space['min_samples_leaf'],
                                         min_samples_split=space['min_samples_split'],
                                         n_estimators=space['n_estimators']
                                        )
            accuracy = cross_val_score(model, X_train, y_train, cv=3).mean()
            return {'loss': -accuracy, 'status': STATUS_OK}

        trials = Trials()
        result = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
        est = {0: 100, 1: 300, 2: 500, 3: 800, 4: 1000, 5: 1200}

        randomforest = RandomForestRegressor(
            max_depth=result['max_depth'],
            max_features=feat[result['max_features']],
            min_samples_leaf=result['min_samples_leaf'],
            min_samples_split=result['min_samples_split'],
            n_estimators=est[result['n_estimators']]
        ).fit(X_train, y_train)

        return randomforest
    optimized_model = tpe_optimization(X_train, y_train, space)
    y_pred = optimized_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate and print the R-squared value
    r_squared = r2_score(y_test, y_pred)
    return y_pred, rmse, r_squared



def pbtree_hyperparameter_tuning():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from hyperopt import hp, fmin, tpe, STATUS_OK

    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(space):
        model = RandomForestRegressor(
            max_depth=space['max_depth'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split'],
            n_estimators=space['n_estimators']
        )
        
        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return {'loss': rmse, 'status': STATUS_OK, 'r2': r2}

    best_hyperparameters = None
    best_rmse = float('inf')
    best_r2 = -float('inf')
    current_population = []

    for _ in range(100):
        new_hyperparameters = {'max_depth': np.random.choice(range(1, 121)),
                               'max_features': np.random.choice(['sqrt', 'log2', None]),
                               'min_samples_leaf': np.random.uniform(0, 0.5),
                               'min_samples_split': np.random.uniform(0, 1),
                               'n_estimators': np.random.choice([100, 300, 500, 800, 1000, 1200])}

        result = objective(new_hyperparameters)
        rmse = result['loss']
        r2 = result['r2']

        if rmse < best_rmse and r2 > best_r2:
            best_rmse = rmse
            best_r2 = r2
            best_hyperparameters = new_hyperparameters

        current_population.append({'hyperparameters': new_hyperparameters, 'rmse': rmse, 'r2': r2})

        if np.random.random() < 0.2:
            exploit_idx = np.random.choice(len(current_population))
            exploit_hyperparameters = current_population[exploit_idx]['hyperparameters']

            for key in new_hyperparameters:
                if np.random.random() < 0.5:
                    new_hyperparameters[key] = exploit_hyperparameters[key]

            result = objective(new_hyperparameters)
            rmse = result['loss']
            r2 = result['r2']

            if rmse < best_rmse and r2 > best_r2:
                best_rmse = rmse
                best_r2 = r2
                best_hyperparameters = new_hyperparameters

            current_population[exploit_idx] = {'hyperparameters': new_hyperparameters, 'rmse': rmse, 'r2': r2}

    return best_hyperparameters, best_rmse, best_r2



time_diff_arr=[]
pbtree_start_time = datetime.now()
print(pbtree_hyperparameter_tuning())
pbtree_end_time = datetime.now()
time_diff1= pbtree_end_time - pbtree_start_time
time_diff_arr.append(str(time_diff1)+'pbtree')


bayesian_opt_start_time=datetime.now()
print(bayesian_opt1())
bayesian_opt_end_time=datetime.now()
time_diff2= bayesian_opt_end_time - bayesian_opt_start_time
time_diff_arr.append(str(time_diff2)+'bayesian_opt')


random_search_start_time=datetime.now()
print(random_search())
random_search_end_time=datetime.now()
time_diff3=random_search_end_time - random_search_start_time
time_diff_arr.append(str(time_diff3)+'random_search')

hyperband_start_time = datetime.now()
print(hyperband())
hyperband_end_time = datetime.now()
time_diff4 = hyperband_end_time - hyperband_start_time
time_diff_arr.append(str(time_diff_arr)+'hyperband')

optimize_random_forest_start_time= datetime.now()
print(optimize_random_forest())
optimize_random_forest_end_time= datetime.now()
time_diff5 = optimize_random_forest_end_time - optimize_random_forest_start_time
time_diff_arr.append(str(time_diff5)+'optimize_random_forest')

with open("time_diff_arr.txt", "w") as file:
    for item in time_diff_arr:
        file.write("%s\n" % item)