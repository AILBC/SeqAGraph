import os
import re
import requests
import zipfile
import numpy as np

from typing import Optional
from rdkit import Chem

PATTERN = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
SMI_PATTERN = r'[A-Za-z]'
REGEX = re.compile(PATTERN)
SMI_REGEX = re.compile(SMI_PATTERN)

def canonicalize_smiles(
    smi: str,
    retain_smi=False,
    map_clear=True,
    cano_with_heavy_atom=True
):
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        cano_smi = smi if retain_smi else ''
        valid = False
    else:
        valid = True
        if mol.GetNumHeavyAtoms() < 2 and cano_with_heavy_atom:
            cano_smi = 'CC'
        elif map_clear:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    if retain_smi:
        return cano_smi, valid
    else:
        return cano_smi

def smi2token(smi: str, return_list: Optional[bool]=False) -> str | list[str]:
    tokens = [token for token in REGEX.findall(smi)]
    if return_list: return tokens
    else: return " ".join(tokens)

def download(url, save_dir, file_name):
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_path):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for i in r.iter_content(chunk_size=128):
                f.write(i)

    with zipfile.ZipFile(save_path) as f:
        f.extractall(path=save_dir)

def smi2augment(
    token: list[str],
    augment_N: Optional[int]=2,
    max_idx: Optional[int]=10,
    only_shuffle: Optional[bool]=False,
    ensure_augment: Optional[bool]=True
):
    smi = ''.join(token).split('.')
    if augment_N > 1:
        smi_list = [[] for _ in range(augment_N - 1)]
        smi_aug_idx = [[] for _ in range(augment_N - 1)]
        smi_shuffle_idx = [[] for _ in range(augment_N - 1)]
        smi_graph = [Chem.MolFromSmiles(_) for _ in smi]
        smi_graph_len = [len(_.GetAtoms()) for _ in smi_graph]
        max_len_idx = np.array(smi_graph_len).argmax(0)
        
        for idx, (s, sgraph) in enumerate(zip(smi, smi_graph)):
            atom_idx = [atom.GetIdx() for atom in sgraph.GetAtoms()]
            if idx == max_len_idx and not only_shuffle:
                if len(atom_idx) < 2:
                    for i in range(augment_N - 1):
                        smi_list[i].append(s)
                        smi_aug_idx[i].append(0)
                else:
                    if ensure_augment:
                        atom_idx = atom_idx[1:max_idx] # ignore 0, which is cano smiles itself
                    else:
                        atom_idx = atom_idx[:max_idx]
                    permute_smi = np.random.choice(atom_idx, augment_N - 1, replace=True if len(atom_idx) < augment_N - 1 else False)
                    for i in range(augment_N - 1):
                        smi_list[i].append(Chem.MolToSmiles(sgraph, rootedAtAtom=int(permute_smi[i])))
                        smi_aug_idx[i].append(int(permute_smi[i]))
            else:
                for i in range(augment_N - 1):
                    smi_list[i].append(s)
        for i in range(augment_N - 1):
            # if len(smi_list[i]) > 1 and ((augment_N == 2) or (np.random.rand() < 0.5)):
            if len(smi_list[i]) > 1 and np.random.rand() < 0.5:
                smi_list[i] = smi_list[i][::-1]
                smi_aug_idx[i] = smi_aug_idx[i][::-1]
                smi_shuffle_idx[i].append(1)
            else:
                smi_shuffle_idx[i].append(0)
            smi_list[i] = '.'.join(smi_list[i])
            smi_list[i] = [t for t in REGEX.findall(smi_list[i])]
    else:
        smi_list = []
        smi_aug_idx = []
        smi_shuffle_idx =[]
    smi_list.insert(0, token)
    smi_aug_idx.insert(0, [0])
    smi_shuffle_idx.insert(0, [0])
    no_augment = sum([sum(_) for _ in smi_aug_idx]) == 0
    return smi_list, smi_aug_idx, smi_shuffle_idx, no_augment

def smigraph_align(
    token: list[str],
    graph: Chem.rdchem.Mol
):
    align_bool = np.array([bool(re.search(SMI_PATTERN, t)) for t in token], dtype=bool)
    graph_idx = np.array([atom.GetIdx() for atom in graph.GetAtoms()], dtype=np.int16)
    align_idx = np.where(align_bool == True)[0]
    if len(graph_idx) == len(align_idx):
        return align_idx
    else:
        return align_idx[:min(len(graph_idx), len(align_idx))]


if __name__ == '__main__':
    import random
    seed = 17
    random.seed(seed)
    np.random.seed(seed)

    # prod = "Brc1cccc(-c2ccccn2)c1"
    # reac = "Brc1ccccn1.OB(O)c1cccc(Br)c1"
    # prod = [token for token in REGEX.findall(prod)]
    # reac = [token for token in REGEX.findall(reac)]
    # prod_list, prod_aug, prod_suf, _ = smi2augment(prod)
    # reac_list, reac_aug, reac_suf, _ = smi2augment(reac)
    # prod_list = [''.join(_) for _ in prod_list]
    # reac_list = [''.join(_) for _ in reac_list]
    # pass