import os
import collections
import numpy as np

from typing import Optional
from tqdm import tqdm
from .dataset_base import DataLoad, DATA_DIR
from .data_utils import smi2augment


class Vocabulary():
    def __init__(self):
        data_50k = Tokenizer('uspto_50k', None)
        data_MIT = Tokenizer('uspto_MIT', None)
        data_full = Tokenizer('uspto_full', None)
        self.data = {'uspto_50k': data_50k, 'uspto_MIT': data_MIT, 'uspto_full': data_full}
        self._generate_vocab()
    
    def _generate_vocab(
        self,
    ):
        token_list = {'uspto_50k': [], 'uspto_MIT': [], 'uspto_full': []}
        for key in token_list.keys():
            if not os.path.exists(os.path.join(DATA_DIR, f'vocabulary({key}).txt')):
                for value in self.data[key].data.values():
                    for reac, prod in zip(value['reactant_token'] ,value['product_token']):
                        reac, prod = reac.split(), prod.split()
                        token_list[key].extend(reac)
                        token_list[key].extend(prod)
                token_collect = sorted(collections.Counter(token_list[key]).items(),
                    key=lambda x: x[1], reverse=True)
                with open(os.path.join(DATA_DIR, f'vocabulary({key}).txt'), 'w') as f:
                    for token, count in token_collect:
                        f.writelines('{token}\t{count}\n'.format(token=token, count=count))
        
        if not os.path.exists(os.path.join(DATA_DIR, f'vocabulary(all).txt')):
            glob_token_list = token_list['uspto_50k'] + token_list['uspto_MIT'] + token_list['uspto_full']
            glob_token_collect = sorted(collections.Counter(glob_token_list).items(),
                key=lambda x: x[1], reverse=True)
            with open(os.path.join(DATA_DIR, f'vocabulary(all).txt'), 'w') as f:
                for token, count in glob_token_collect:
                    f.writelines('{token}\t{count}\n'.format(token=token, count=count))
        


class Tokenizer(DataLoad):
    def __init__(
        self,
        dataset_name: str,
        vocab_name='all'
    ):
        super(Tokenizer, self).__init__(
            dataset_name=dataset_name,
            preprocess_step='tokenize'
        )
        self.load_vocab(vocab_name=vocab_name)
        self.load()
    
    def tokenize(
        self,
        augment_N: Optional[int]=2,
        max_perm_idx: Optional[int]=10,
    ):
        max_len = 0
        avg_len = 0
        total_reaction = 0
        total_raw_reaction = 0
        self.logger.info(f'{self.preprocess_step}--> start tokenize with vocabulary({self.vocab_name}).txt')
        for mode, data in self.data.items():
            tokenize_result = {'reaction_type': [], 'reactant': [], 'product': [], 'reaction_length': [],\
                               'reac_aug': [], 'prod_aug': [], 'reac_suf': [], 'prod_suf': []}
            for reac, prod, rtype in tqdm(zip(data['reactant_token'], data['product_token'], data['reaction_type']),
                desc=f'tokenize {self.name}: token({mode}).csv...', total=len(data['reactant_token'])):
                t_reac, t_prod = reac.split(), prod.split()
                t_reac, t_reac_augidx, t_reac_sufidx, no_augment = smi2augment(t_reac, augment_N if mode == 'train' else 1, max_perm_idx)
                t_prod, t_prod_augidx, t_prod_sufidx, no_augment = smi2augment(t_prod, augment_N if mode == 'train' else 1, max_perm_idx)

                for i in range(len(t_reac)):
                    if len(t_reac[i]) > 0 and len(t_prod[i]) > 0:
                        t_reac[i], t_prod[i] = [self.vocabulary.get(i) for i in t_reac[i]],\
                            [self.vocabulary.get(i) for i in t_prod[i]]
                        total_reaction += 1
                        if i == 0: total_raw_reaction += 1
                tokenize_result['reactant'].append(t_reac)
                tokenize_result['product'].append(t_prod)
                reaction_len = len(t_reac[0]) + len(t_prod[0])
                tokenize_result['reaction_length'].append(reaction_len)
                tokenize_result['reaction_type'].append(rtype)
                tokenize_result['reac_aug'].append(t_reac_augidx)
                tokenize_result['prod_aug'].append(t_prod_augidx)
                tokenize_result['reac_suf'].append(t_reac_sufidx)
                tokenize_result['prod_suf'].append(t_prod_sufidx)
                max_len = reaction_len if reaction_len > max_len else max_len
                avg_len += reaction_len
        
            np.savez(os.path.join(self.process_dir, f'tokenize({mode})({self.vocab_name}).npz'), tokenize_data=tokenize_result)
        self.logger.info('{step}--> maximun length:{max_len}'.format(step=self.preprocess_step, max_len=max_len))
        self.logger.info('{step}--> average length:{avg_len:.6}'.format(step=self.preprocess_step, avg_len=avg_len / total_reaction))
        self.logger.info('{step}--> total reaction:{total_reaction}'.format(step=self.preprocess_step, total_reaction=total_reaction))
        self.logger.info('{step}--> total raw reaction:{total_raw_reaction}'.format(step=self.preprocess_step, total_raw_reaction=total_raw_reaction))
        self.logger.info(f'{self.preprocess_step}--> finish.')
        self.logger.info('\n')

if __name__ == '__main__':
    # vocab = Vocabulary()
    # del vocab
    tokenizer = Tokenizer(
        dataset_name='uspto_50k',
        vocab_name='uspto_50k'
    )
    tokenizer.tokenize(2)