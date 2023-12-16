import os
import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))

for mode in ['train', 'eval', 'test']:
    data_path = os.path.join(DIR, f'{mode}.txt')
    with open(data_path, 'r') as raw_data:
        data = {'class': [], 'reactants>reagents>production': []}
        for smi in raw_data:
            if len(smi) <= 1: continue
            data['class'].append(0)
            data['reactants>reagents>production'].append(smi.strip())
        pd.DataFrame(data).to_csv(os.path.join(DIR, f'{mode}.csv'), index=False)