import os
import pickle
import random
import torch
import numpy as np

from copy import deepcopy as dcopy
from .dataset_base import DataLoad

class Split_Shuffle(DataLoad):
    def __init__(
        self,
        dataset_name: str,
        vocab_name='all',
        mode='eval',
        K='4',
        kernel='spd',
        seed=17
    ):
        super(Split_Shuffle, self).__init__(
            dataset_name=dataset_name,
            preprocess_step='split_shuffle'
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.load_vocab(vocab_name=vocab_name)
        self.load(
            mode=mode,
            K=K,
            kernel=kernel
        )
        self.mode = mode
        self.K = K
        self.kernel = kernel
    
    def split_shuffle(
        self,
        split_num=1e3
    ):
        split_num = int(split_num)
        shuffle_idx = np.random.permutation(len(self.data[self.mode]))[:split_num]
        split_data = dcopy(self.data[self.mode])
        split_data = split_data[shuffle_idx]
        
        data_path = os.path.join(self.process_dir, f'K_{self.K}({self.kernel})')
        # np.savez(os.path.join(data_path, f'featurize({self.mode})({self.vocab_name})-s{split_num}.npz'),
        #          featurize_result=split_data)
        with open(os.path.join(data_path, f'featurize({self.mode})({self.vocab_name})-s{split_num}.pkl'), 'wb') as f:
            pickle.dump(split_data, f)
        
        self.logger.info(f'{self.preprocess_step}--> featurize({self.mode})({self.vocab_name})-s{split_num}.pkl split&shuffle finish.')
        self.logger.info(f'{self.preprocess_step}--> finish.')
        self.logger.info('\n')

if __name__ == '__main__':
    split_shuffle = Split_Shuffle(
        dataset_name='uspto_50k'
    )
    split_shuffle.split_shuffle()