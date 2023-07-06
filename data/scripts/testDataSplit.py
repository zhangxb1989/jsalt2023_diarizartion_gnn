import pickle

import numpy as np
from h5py import Dataset, Group, File
from sklearn.model_selection import train_test_split

with File('C:/Users/ALLEN/Desktop/voxceleb2_clean.h5', 'r') as f:
    # print(f.keys())
    # print(f['Data'][()].shape)
    # print(f['Name'][()].shape)

    string_list = f['Name'][()]

    # String to Integer Mapping
    integer_list = [int(item[0].decode('utf-8').split('id')[1]) for item in string_list]
    numpyNameList = np.array(integer_list, dtype=np.int16)

    X = f['Data'][()] #features
    y = numpyNameList #labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_lst = [X_train, y_train]
    test_lst = [X_test, y_test]

    with open("voxceleb_train.pkl", "wb") as f:
         pickle.dump(train_lst, f)

    with open("voxceleb_test.pkl", "wb") as f:
         pickle.dump(test_lst, f)

