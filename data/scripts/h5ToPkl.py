import pickle

import numpy as np
from h5py import Dataset, Group, File

with File('C:/Users/ALLEN/Desktop/voxceleb2_clean.h5', 'r') as f:

    string_list = f['Name'][()]

    integer_list = [int(item[0].decode('utf-8').split('id')[1]) for item in string_list]
    numpyNameList = np.array(integer_list, dtype=np.int16)

    lst = [f['Data'][()], numpyNameList]

    with open("voxceleb.pkl", "wb") as f:
        pickle.dump(lst, f)
