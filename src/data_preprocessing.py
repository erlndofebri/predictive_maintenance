# Import library
import utils as utils
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



# Import fungsi-fungsi
def split_num_cat(X_train, num_cols, cat_cols):
    # Split data
    X_train_num = X_train[num_cols]
    X_train_cat = X_train[cat_cols]

    # Validasi
    print('Data numerik shape   :', X_train_num.shape)
    print('Data kategorik shape :', X_train_cat.shape)

    return X_train_num, X_train_cat

def fit_num_imputer(X_train_num, num_imputer_path):
    # Buat imputer
    imputer_num = SimpleImputer(missing_values = np.nan,
                                strategy = 'median')
    
    # Fit imputer
    imputer_num.fit(X_train_num)

    # Dump imputer
    utils.pickle_dump(imputer_num, num_imputer_path)

    return imputer_num

def transform_num_imputer(X_num, num_imputer):
    # Hard copy data
    X_num = X_num.copy()

    # Transfrom
    X_num_imputed = pd.DataFrame(
        num_imputer.transform(X_num),
        columns = X_num.columns,
        index = X_num.index
    )

    # Validasi
    print('Data shape :', X_num_imputed.shape)
    print('')
    print('Missing val:\n', X_num_imputed.isnull().sum())
    print('')

    return X_num_imputed

def fit_cat_imputer(X_train_cat, cat_imputer_path):
    # Buat imputer
    imputer_cat = SimpleImputer(missing_values = np.nan,
                                strategy = 'constant',
                                fill_value = 'KOSONG')  # isi dengan KOSONG
    
    # Fit imputer
    imputer_cat.fit(X_train_cat)

    # Dump imputer
    utils.pickle_dump(imputer_cat, cat_imputer_path)

    return imputer_cat

def transform_cat_imputer(X_cat, cat_imputer):
    # Hard copy data
    X_cat = X_cat.copy()

    # Transfrom
    X_cat_imputed = pd.DataFrame(
        cat_imputer.transform(X_cat),
        columns = X_cat.columns,
        index = X_cat.index
    )

    # Validasi
    print('Data shape :', X_cat_imputed.shape)
    print('')
    print('Missing val:\n', X_cat_imputed.isnull().sum())
    print('')

    return X_cat_imputed

def cat_encoding(X_cat):
    # Mapping function
    map_dict = {
        'L': 0,
        'M': 1,
        'H': 2
    }

    # Fungsi
    X_cat_enc = X_cat.copy()
    X_cat_enc['Type'] = X_cat['Type'].map(map_dict)

    # Validasi
    print('Data shape:', X_cat_enc.shape)

    return X_cat_enc

def concat_data(X_num, X_cat):
    X_concat = pd.concat([X_num, X_cat], axis=1)

    # Validasi
    print('Data shape:', X_concat.shape)

    return X_concat

def fit_scaler(X_train, scaler_path):
    # Buat scaler
    scaler = StandardScaler()

    # Fit scaler
    scaler.fit(X_train)

    # Dump
    utils.pickle_dump(scaler, scaler_path)

    return scaler

def transform_scaler(X, scaler):
    X_clean = pd.DataFrame(
        scaler.transform(X),
        columns = X.columns,
        index = X.index
    )

    # Validasi
    print('Data shape:', X_clean.shape)

    return X_clean

def preprocess_data(X, types, CONFIG_DATA):
    if X is None:
        # Load data
        path = f'{types}_set_path'
        X = utils.pickle_load(CONFIG_DATA[path][0])

    # Lakukan preprocessing
    # Pertama, split data
    num_cols = CONFIG_DATA['num_cols']
    cat_cols = CONFIG_DATA['cat_cols']
    X_num, X_cat = split_num_cat(X, num_cols, cat_cols)

    # Lakukan imputasi
    if types=='train':
        # Kalo data train, buat preprocessor
        num_imputer_path = CONFIG_DATA['num_imputer_path']
        cat_imputer_path = CONFIG_DATA['cat_imputer_path']
        num_imputer = fit_num_imputer(X_num, num_imputer_path)
        cat_imputer = fit_cat_imputer(X_cat, cat_imputer_path)
    else:
        # Kalo bukan train, load preprocessor
        num_imputer_path = CONFIG_DATA['num_imputer_path']
        cat_imputer_path = CONFIG_DATA['cat_imputer_path']
        num_imputer = utils.pickle_load(num_imputer_path)
        cat_imputer = utils.pickle_load(cat_imputer_path)
        
    X_num_imputed = transform_num_imputer(X_num, num_imputer)
    X_cat_imputed = transform_cat_imputer(X_cat, cat_imputer)

    # Lakukan encoding
    X_cat_enc = cat_encoding(X_cat_imputed)

    # Lakukan concat data
    X_concat = concat_data(X_num_imputed, X_cat_enc)

    # Lakukan scaling
    if types=='train':
        # Kalo data train, buat scaler
        scaler_path = CONFIG_DATA['scaler_path']
        scaler = fit_scaler(X_concat, scaler_path)
    else:
        # Kalo bukan train, load scaler
        scaler_path = CONFIG_DATA['scaler_path']
        scaler = utils.pickle_load(scaler_path)

    X_clean = transform_scaler(X_concat, scaler)

    # Validasi
    print('Data shape:', X_clean.shape)

    # Dump file
    if types in ['train', 'valid', 'test']:
        clean_path = CONFIG_DATA[f'{types}_clean_path']
        utils.pickle_dump(X_clean, clean_path)

    return X_clean
    

# Program utama
if __name__ == "__main__":
    # 1. Load config file
    CONFIG_DATA = utils.config_load()

    # 2. Preprocess data
    preprocess_data(X=None, types='train', CONFIG_DATA=CONFIG_DATA)
    preprocess_data(X=None, types='valid', CONFIG_DATA=CONFIG_DATA)
    preprocess_data(X=None, types='test', CONFIG_DATA=CONFIG_DATA)
