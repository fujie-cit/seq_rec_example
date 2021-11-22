import pandas as pd
import pickle
from torch.utils.data import DataLoader
import torch

INFO_CSV_PATH='../data/info.csv'
PICKLE_PATH='../data/OGVC_LLD_PICKLE/data_norm.pkl'

def load_data():
    info_df = pd.read_csv(INFO_CSV_PATH, index_col=0)
    data_dict = dict(pickle.load(open(PICKLE_PATH, 'rb')))

    data = dict()
    for flag_column in ['is_train', 'is_vali', 'is_test']:
        data_list = []
        df_sub = info_df[info_df[flag_column] == True]
        for idx in df_sub.index:
            x = data_dict[idx]
            y = df_sub['y_number'][idx]
            data_list.append((x, y))
        data[flag_column] = data_list
    
    data_train = data['is_train']
    data_vali = data['is_vali']
    data_test = data['is_test']

    return data_train, data_vali, data_test

def collate_seq(data_list):
    # x, lengths, y
    
    max_length = 0
    x_list = []
    len_list = []
    y_list = []
    for x, y in data_list:
        xlen = len(x)
        if xlen > max_length:
            max_length = xlen
        x_list.append(x)
        len_list.append(xlen)
        y_list.append(y)

    num_data = len(data_list)
    dim_features = data_list[0][0].shape[1]

    x_tensor = torch.zeros((max_length, num_data, dim_features))
    for i, (x, _) in enumerate(data_list):
        xlen = len(x)
        x_tensor[:xlen, i, :] = torch.tensor(x)

    len_tensor = torch.tensor(len_list)
    y_tensor = torch.tensor(y_list)

    return x_tensor, len_tensor, y_tensor

