import os
import pickle
import logging
import pandas as pd
import numpy as np

from abc import ABCMeta
from typing import Optional

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')

class Dataset(metaclass=ABCMeta):
    """
    baisc class for dataset.
    """

    def __init__(
        self,
        dataset_name: str
    ):
        super(Dataset, self).__init__()
        self.name = dataset_name
        self.data_dir = os.path.join(DATA_DIR, self.name)
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.process_dir = os.path.join(self.data_dir, 'process')
        self._create_directory()

    def _create_directory(self):
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)
        if not os.path.exists(self.process_dir):
            os.mkdir(self.process_dir)


class DataLoad(Dataset):
    """
    basic class for preprocess data load.
    """

    def __init__(
        self,
        dataset_name: str,
        preprocess_step: str,
        use_split=False,
        split_name: Optional[str]=''
    ):
        super(DataLoad, self).__init__(
            dataset_name=dataset_name
        )
        self.preprocess_step = preprocess_step
        self.use_split = use_split
        self.split_name = split_name
        self.data = {'train': None, 'eval': None, 'test': None}
        self.vocab_name = None
        self.vocabulary = None

        self.logger = logging.getLogger()
        log_file = logging.FileHandler(os.path.join(self.process_dir, 'preprocess.log'))
        log_format = logging.Formatter('%(asctime)s %(message)s')
        log_file.setFormatter(log_format)
        self.logger.addHandler(log_file)
        self.logger.setLevel('INFO')
        self.logger.info(f'Dataset:{self.name}; Preprocess:{self.preprocess_step}')
    
    def load(
        self,
        mode='',
        K=4,
        kernel='spd'
    ):
        if mode == '':
            self.data = {'train': self._load('train'),
                         'eval': self._load('eval'),
                         'test': self._load('test')}
        elif mode in ['train', 'eval', 'test']:
            self.data[mode] = self._load(mode, K, kernel)

    def _load(
        self,
        mode: str,
        K: Optional[int] = 4,
        kernel: Optional[str] = 'spd'
    ):
        if self.preprocess_step == 'smi2token':
            data_path = os.path.join(self.raw_dir, f'{mode}.csv')
            data = pd.read_csv(data_path)
        elif self.preprocess_step == 'tokenize':
            data_path = os.path.join(self.process_dir, f'token({mode}).csv')
            data = pd.read_csv(data_path)
        elif self.preprocess_step == 'featurize':
            data_path = os.path.join(self.process_dir, f'tokenize({mode})({self.vocab_name}).npz')
            data = np.load(data_path, allow_pickle=True)['tokenize_data'].tolist()
        elif self.preprocess_step == 'split_shuffle':
            data = 0
            data_path = os.path.join(self.process_dir, f'K_{K}({kernel})')
            # data_list = np.load(os.path.join(data_path, f'featurize({mode})({self.vocab_name})-record.npz'),
            #                     allow_pickle=True)['record'].tolist()
            with open(os.path.join(data_path, f'featurize({mode})({self.vocab_name})-record.pkl'), 'rb') as f:
                data_list = pickle.load(f)
            for fl_name in data_list.keys():
                data_path = os.path.join(data_path, fl_name)
                if os.path.exists(data_path):
                    if isinstance(data, int):
                        with open(data_path, 'rb') as f:
                            data = pickle.load(f)
                        # data = np.load(data_path, allow_pickle=True)['featurize_result']
                    else:
                        with open(data_path, 'rb') as f:
                            next_data = pickle.load(f)
                        # next_data = np.load(data_path, allow_pickle=True)['featurize_result']
                        data = np.concatenate([data, next_data], axis=0)
        return data

    def load_vocab(
        self,
        vocab_name='all'
    ):
        if vocab_name != None:
            vocab_path = os.path.join(DATA_DIR, f'vocabulary({vocab_name}).txt')
            token, count = [], []
            with open(vocab_path, 'r') as f:
                for data in f:
                    t, c = data.strip().split('\t')
                    token.append(t)
                    count.append(int(c))
            self.vocab_name = vocab_name
            self.vocabulary = {t:id for id, t in enumerate(token)}
