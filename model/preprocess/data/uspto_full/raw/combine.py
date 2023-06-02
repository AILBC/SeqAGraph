import os
import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))

for mode in ['train', 'eval', 'test']:
    prod_path, reac_path = os.path.join(DIR, f'src-{mode}.txt'), os.path.join(DIR, f'tgt-{mode}.txt')
    with open(prod_path, 'r') as prod_data:
        with open(reac_path, 'r') as reac_data:
            data = {'class': [], 'reactants>reagents>production': []}
            for prod_smi, reac_smi in zip(prod_data, reac_data):
                if len(prod_smi) <= 1 or len(reac_smi) <= 1: continue
                data['class'].append(0)
                data['reactants>reagents>production'].append(reac_smi.strip() + '>>' + prod_smi.strip())
            pd.DataFrame(data).to_csv(os.path.join(DIR, f'{mode}.csv'), index=False)