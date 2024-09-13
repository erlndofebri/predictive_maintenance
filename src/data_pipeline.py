# Import library yang dibutuhkan
import pandas as pd
import utils as utils
from sklearn.model_selection import train_test_split


# Paste seluruh fungsi yang dibuat
def read_data():
    # Read data
    data_path = CONFIG_DATA['raw_dataset_path']
    data = pd.read_csv(data_path)
    print('Data shape       :', data.shape)

    # Cek apa ada duplikat
    print('Duplikat data    :', data.duplicated().sum())

    # Drop duplikat
    data = data.drop_duplicates()

    # Validasi hasil
    print('Data shape final :', data.shape)

    # Simpan hasil dalam pickle
    dump_path = CONFIG_DATA['dataset_path']
    utils.pickle_dump(data, dump_path)

    return data

def split_input_output():
    # Load data otomatis
    dataset_path = CONFIG_DATA['dataset_path']
    data = utils.pickle_load(dataset_path)
    
    # Buat data output, y
    output_cols = CONFIG_DATA['output_cols']
    y = data[output_cols]
    y = y.apply(lambda types: 0 if types=="No Failure" else 1)
    
    # Buat data input, X
    drop_cols = CONFIG_DATA['drop_cols']
    X = data.drop(columns=drop_cols, axis=1)

    # Validasi
    print('Input shape   :', X.shape)
    print('Output shape  :', y.shape)

    # Dump file
    dump_path_input = CONFIG_DATA['input_set_path']
    dump_path_output = CONFIG_DATA['output_set_path']
    dump_path_input_cols = CONFIG_DATA['input_cols_path']
    utils.pickle_dump(X, dump_path_input)
    utils.pickle_dump(y, dump_path_output)
    utils.pickle_dump(X.columns, dump_path_input_cols)
    
    return X, y

def split_train_test():
    # Load data X dan y
    input_path = CONFIG_DATA['input_set_path']
    output_path = CONFIG_DATA['output_set_path']
    X = utils.pickle_load(input_path)
    y = utils.pickle_load(output_path)

    # Train test split
    test_size = CONFIG_DATA['test_size']
    random_state = CONFIG_DATA['seed']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size = test_size,
                                                        random_state = random_state)

    # Train valid split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        stratify=y_train,
                                                        test_size = test_size,
                                                        random_state = random_state)
    
    # Validasi
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump file
    xtrain_path = CONFIG_DATA['train_set_path'][0]
    ytrain_path = CONFIG_DATA['train_set_path'][1]
    xvalid_path = CONFIG_DATA['valid_set_path'][0]
    yvalid_path = CONFIG_DATA['valid_set_path'][1]
    xtest_path = CONFIG_DATA['test_set_path'][0]
    ytest_path = CONFIG_DATA['test_set_path'][1]
    utils.pickle_dump(X_train, xtrain_path)
    utils.pickle_dump(y_train, ytrain_path)
    utils.pickle_dump(X_valid, xvalid_path)
    utils.pickle_dump(y_valid, yvalid_path)
    utils.pickle_dump(X_test, xtest_path)
    utils.pickle_dump(y_test, ytest_path)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# Buat program eksekusi
if __name__ == '__main__':
    # Load config data
    CONFIG_DATA = utils.config_load()

    # Read all data & split data
    read_data()
    split_input_output()
    split_train_test()