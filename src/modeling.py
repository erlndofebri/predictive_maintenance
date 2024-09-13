# Masukkan library
import utils as utils
import copy as copy
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# Masukkan fungsi
def create_model_param():
    """Create the model objects"""
    knn_params = {
        'n_neighbors': [50, 100, 200],
    }

    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, None]
    }
    
    lgr_params = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1],
        'max_iter': [100, 300, 500]
    }

    rf_params = {
        'n_estimators': [100, 200, 300]
    }

    # Create model params
    list_of_param = {
        'KNeighborsClassifier': knn_params,
        'DecisionTreeClassifier': dt_params,
        'LogisticRegression': lgr_params,
        'RandomForestClassifier': rf_params
    }

    return list_of_param

def create_model_object():
    """Create the model objects"""
    print("Creating model objects")

    # Create model objects
    knn = KNeighborsClassifier()
    lgr = LogisticRegression(solver='liblinear')
    dt = DecisionTreeClassifier(random_state=123)
    rf = RandomForestClassifier(random_state=123)

    # Create list of model
    list_of_model = [
        {'model_name': knn.__class__.__name__, 'model_object': knn},
        {'model_name': lgr.__class__.__name__, 'model_object': lgr},
        {'model_name': dt.__class__.__name__, 'model_object': dt},
        {'model_name': rf.__class__.__name__, 'model_object': rf}
    ]

    return list_of_model

def train_model():
    # Load dataset
    # Hanya menggunakan data train & valid
    X_train = utils.pickle_load(CONFIG_DATA['train_clean_path'])
    y_train = utils.pickle_load(CONFIG_DATA['train_set_path'][1])
    X_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][1])
    
    # Create list of params & models
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])
        model_param = list_of_param[model_name]

        # Debug message
        print('Training model :', model_name)

        # Create model object
        model = GridSearchCV(estimator = model_obj,
                             param_grid = model_param,
                             cv = 5,
                             verbose=10,
                             scoring = 'roc_auc')
        
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        
        # Get score
        train_score = roc_auc_score(y_train, y_pred_proba_train)
        valid_score = roc_auc_score(y_valid, y_pred_proba_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model,
            'train_auc': train_score,
            'valid_auc': valid_score,
            'best_params': model.best_params_
        }

        print("Done training")
        print("")

    # Dump data
    utils.pickle_dump(list_of_param, CONFIG_DATA['list_of_param_path'])
    utils.pickle_dump(list_of_model, CONFIG_DATA['list_of_model_path'])
    utils.pickle_dump(list_of_tuned_model, CONFIG_DATA['list_of_tuned_model_path'])

    return list_of_param, list_of_model, list_of_tuned_model    

def get_best_model():
    # Load tuned model
    list_of_tuned_model = utils.pickle_load(CONFIG_DATA['list_of_tuned_model_path'])

    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = -99999
    best_model_param = None

    for model_name, model in list_of_tuned_model.items():
        if model['valid_auc'] > best_performance:
            best_model_name = model_name
            best_model = model['model']
            best_performance = model['valid_auc']
            best_model_param = model['best_params']

    # Dump the best model
    utils.pickle_dump(best_model, CONFIG_DATA['best_model_path'])

    # Print
    print('=============================================')
    print('Best model        :', best_model_name)
    print('Metric score      :', best_performance)
    print('Best model params :', best_model_param)
    print('=============================================')

    return best_model

def get_best_threshold():
    # Load data & model
    X_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][1])
    best_model = utils.pickle_load(CONFIG_DATA['best_model_path'])

    # Get the proba pred
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Initialize
    metric_threshold = pd.Series([])
    
    # Optimize
    for threshold_value in THRESHOLD:
        # Get predictions
        y_pred = (y_pred_proba >= threshold_value).astype(int)

        # Get the F1 score
        metric_score = recall_score(y_valid, y_pred, average='weighted')

        # Add to the storage
        metric_threshold[metric_score] = threshold_value

    # Find the threshold @max metric score
    metric_score_max_index = metric_threshold.index.max()
    best_threshold = metric_threshold[metric_score_max_index]
    print('=============================================')
    print('Best threshold :', best_threshold)
    print('Metric score   :', metric_score_max_index)
    print('=============================================')
    
    # Dump file
    utils.pickle_dump(best_threshold, CONFIG_DATA['best_threshold_path'])

    return best_threshold



# Panggil program utama
if __name__ == '__main__':
    # 1. Load config
    CONFIG_DATA = utils.config_load()

    # 2. Train & Optimize the model
    train_model()

    # 3. Get the best model
    get_best_model()

    # 4. Get the best threshold for the best model
    THRESHOLD = np.linspace(0, 1, 100)
    get_best_threshold()