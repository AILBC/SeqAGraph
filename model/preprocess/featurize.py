import os
import pickle
import numpy as np

from rdkit import Chem
from typing import Optional
from tqdm import tqdm
from copy import deepcopy as dcopy
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
from .dataset_base import DataLoad
from .data_utils import smigraph_align, smi2token
from .chem_utils import get_atom_feat, get_bond_feat, BOND_FDIM, BOND_TYPE_NUM

def get_feat(
    token: list[int] | str,
    vocab: dict[str:int],
    perm_idx=-1,
    suf_idx=-1,
    dual_task=-1,
    reaction_class=-1,
    max_deg: Optional[int]=9,
    K: Optional[int]=4,
    kernel: Optional[str]='spd',
    max_bond_count: Optional[int]=15, #maximum count for each type of bond
    max_dist_count: Optional[int]=15, #maximum count for each distance
):
    rvocab = {id:t for t, id in vocab.items()}
    atom_feat, bond_idx, bond_feat, bond_type_list, align_idx = get_1hop_feat(
        token=token,
        rvocab=rvocab,
        perm_idx=perm_idx,
        suf_idx=suf_idx,
        dual_task=dual_task,
        reaction_class=reaction_class
    )
    deg_matrix, dist_matrix = get_attention_bias(
        node_num=atom_feat.shape[0],
        bond_idx=bond_idx,
        max_deg=max_deg
    )
    bond_idx, bond_feat, bond_neibor = get_khop_feat(
        node_num=atom_feat.shape[0],
        bond_idx=bond_idx,
        bond_feat=bond_feat,
        bond_type_list=bond_type_list,
        K=K,
        kernel=kernel,
        max_bond_count=max_bond_count,
        max_dist_count=max_dist_count
    )
    graph_feat = {'atom_feat': atom_feat, 'bond_idx': bond_idx, 'bond_feat': bond_feat, 'bond_neibor': bond_neibor,\
                  'deg': deg_matrix, 'dist': dist_matrix, 'align_idx': align_idx}
    return graph_feat
    

def get_1hop_feat(
    token: list[int] | str,
    rvocab: dict[int:str],
    perm_idx=-1,
    suf_idx=-1,
    dual_task=-1,
    reaction_class=-1
):
    if isinstance(token, list):
        token_list = [rvocab.get(i) for i in token]
    elif isinstance(token ,str):
        token_list = smi2token(token, True)
    token = ''.join(token_list)
    mol = Chem.MolFromSmiles(token)
    align_idx = smigraph_align(token_list, mol)

    atom_feat = []
    bond_start = []
    bond_end = []
    bond_feat = []
    bond_type_list = []

    for atom in mol.GetAtoms():
        atom_feat.append(get_atom_feat(
            atom=atom,
            dual_task=dual_task,
            reaction_class=reaction_class,
            perm_idx=perm_idx,
            suf_idx=suf_idx
        ))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_start.extend([start, end])
        bond_end.extend([end, start])
        f_bond, bond_type = get_bond_feat(bond)
        # bond_feat.extend([f_bond, f_bond])
        bond_feat.append(f_bond)
        bond_feat.append(f_bond)
        bond_type_list.extend([bond_type, bond_type])

    atom_feat = np.array(atom_feat, dtype=np.int16)
    bond_feat = np.array(bond_feat, dtype=np.int16)
    bond_type_list = np.array(bond_type_list, dtype=np.int16)
    bond_idx = np.array([bond_start, bond_end], dtype=np.int16)
    perm = (bond_idx[0] * mol.GetNumAtoms() + bond_idx[1]).argsort()
    bond_idx = bond_idx[:, perm]
    bond_feat = bond_feat[perm]
    bond_type_list = bond_type_list[perm]
    return atom_feat, bond_idx, bond_feat, bond_type_list, align_idx

def get_khop_feat(
    node_num: int,
    bond_idx: np.ndarray,
    bond_feat: np.ndarray,
    bond_type_list: np.ndarray,
    K: Optional[int]=4, #maximum hop number for neighbors
    kernel: Optional[str]='spd', #kernel for extract neighbors
    max_bond_count: Optional[int]=15, #maximum count for each type of bond
    max_dist_count: Optional[int]=15, #maximum count for each distance
):
    """
    from 'How Powerful are K-hop Message Passing Graph Neural Networks'
    """
    if len(bond_idx) == 0:
        bond_idx, bond_feat, bond_neibor = [], [], []
        bond_neibor.append(np.array([], dtype=np.int32))
        for i in range(K):
            bond_idx.append(np.array([], dtype=np.int16))
            bond_feat.append(np.array([], dtype=np.int16))
        return bond_idx, bond_feat, bond_neibor
        
    adj_list = get_khop_adj(
        node_num=node_num,
        bond_idx=bond_idx,
        K=K
    )

    #transform adj_list to spd adj_list
    if kernel == 'spd':
        previous_path = dcopy(adj_list[0])
        for i in range(1, len(adj_list)):
            adj_i = dcopy(adj_list[i])
            adj_i[previous_path > 0] = 0
            previous_path = previous_path + adj_i
            adj_list[i] = adj_i

    #generate k-hop bond feature(k>1 use path encoding)
    bond_idx = [dcopy(bond_idx)]
    bond_feat = [dcopy(bond_feat)]
    for i in range(1, len(adj_list)):
        adj_i = dcopy(adj_list[i])
        bond_idx_i = np.array(np.where(adj_i > 0), dtype=np.int16)
        bond_feat_i = adj_i[list(bond_idx_i[0]), list(bond_idx_i[1])]
        bond_feat_i[bond_feat_i > max_bond_count] = max_bond_count
        bond_idx.append(bond_idx_i)
        bond_feat.append(bond_feat_i)
    bond_neibor = get_bond_neighbors(bond_idx, 1)
    return bond_idx, bond_feat, bond_neibor

def get_khop_adj(
    node_num: int,
    bond_idx: np.ndarray,
    K: Optional[int]=4
) -> list[np.ndarray]:
    adj = np.zeros((node_num, node_num), dtype=np.int16)
    adj[list(bond_idx[0]), list(bond_idx[1])] = 1
    adj_list = [dcopy(adj)]
    for i in range(K-1):
        adj_list.append(adj_list[-1] @ adj)
    for i in range(len(adj_list)):
        np.fill_diagonal(adj_list[i], 0)
    return adj_list

def get_attention_bias(
    node_num: int,
    bond_idx: int,
    max_deg: Optional[int]=9
) -> np.ndarray:
    if len(bond_idx) > 0:
        adj_matrix = np.zeros((node_num, node_num), dtype=np.int16)
        adj_matrix[bond_idx[0], bond_idx[1]] = 1
        deg_matrix = adj_matrix.sum(axis=0)
        deg_matrix[deg_matrix > max_deg] = max_deg

        dist_matrix = floyd_warshall(csr_matrix(adj_matrix), directed=False)
        dist_matrix[np.isinf(dist_matrix)] = -1
        dist_matrix = dist_matrix.astype(np.int16)
    else:
        deg_matrix = np.zeros((node_num), dtype=np.int16)
        dist_matrix = np.full((node_num, node_num), fill_value=-1, dtype=np.int16)
    np.fill_diagonal(dist_matrix, 0)
    return deg_matrix, dist_matrix

def get_bond_neighbors(
    bond_idx: list[np.ndarray],
    max_k: Optional[int]=1
) -> list[np.ndarray]:
    neighbor_list = []
    for k in range(max_k):
        idx_k = bond_idx[k]
        if idx_k.shape[-1] < 1: neighbor_list.append(np.array([], dtype=np.int16))
        bond_range = np.arange(idx_k.shape[1], dtype=np.int16)
        src_idx = []
        neighbor_idx = []
        for idx in range(idx_k.shape[1]):
            i, j = idx_k[0][idx], idx_k[1][idx]
            k_idx = (idx_k[1] == i) & (idx_k[0] != j)
            neighbor_num = k_idx.sum(axis=-1)
            neighbor_idx += bond_range[k_idx].tolist()
            src_idx += [idx] * neighbor_num
        src_idx = np.array(src_idx, dtype=np.int16)
        neighbor_idx = np.array(neighbor_idx, dtype=np.int16)
        neighbor_list.append(np.stack([src_idx, neighbor_idx], axis=0))
    return neighbor_list


class Featurize(DataLoad):
    def __init__(
        self,
        dataset_name: str,
        vocab_name='all'
    ):
        super(Featurize, self).__init__(
            dataset_name=dataset_name,
            preprocess_step='featurize'
        )
        self.load_vocab(vocab_name=vocab_name)
        self.load()
    
    def featurize(
        self,
        max_split_count: Optional[int]=1e5,
        max_deg: Optional[int]=9,
        K: Optional[int]=4, #maximum hop number for neighbors
        kernel: Optional[str]='spd', #kernel for extract neighbors
        max_bond_count: Optional[int]=15, #maximum count for each type of bond
        max_dist_count: Optional[int]=15, #maximum count for each distance
    ):
        """
        compute 1hop feature, khop feature, and global attention bias
        Args:
            max_split_count(int)=1e5, maximum number of reaction in each preprocess file
            max_deg(int)=9: maximum degree to consider in attention bias
            K(int)=4: maximum hop for k-hop GNN
            kernel(str)='spd': the kernel for calculating hop, 'spd' for shortest path distance, 'gd' for graph diffusion
            max_bond_count(int)=32: maximum count for each type of bond when computing peripheral subgraph
            max_dist_count(int)=32: maximum count for each distance when computing peripheral subgraph
        """
        self.logger.info(f'{self.preprocess_step}--> start featurize with vocabulary({self.vocab_name}).txt')
        self.featurize_path = os.path.join(self.process_dir, f'K_{K}({kernel})')
        if not os.path.exists(self.featurize_path):
            os.mkdir(self.featurize_path)
        for mode, data in self.data.items():
            split_count = 0
            split_record = {}
            featurize_result = []
            for reac, prod, reaction_class, reaction_len, reac_aug, prod_aug, reac_suf, prod_suf in tqdm(zip(
                data['reactant'], data['product'], data['reaction_type'], data['reaction_length'], data['reac_aug'], data['prod_aug'],
                data['reac_suf'], data['prod_suf']),
                desc=f'featurize {self.name}: tokenize({mode})({self.vocab_name}).npz...', total=len(data['reactant'])):
                reac_graph, prod_graph = [], []
                for i in range(len(reac)):
                    reac_graph.append(get_feat(
                        token=reac[i],
                        vocab=self.vocabulary,
                        perm_idx=reac_aug[i][0],
                        suf_idx=reac_suf[i][0],
                        dual_task=1,
                        reaction_class=reaction_class,
                        max_deg=max_deg,
                        K=K,
                        kernel=kernel,
                        max_bond_count=max_bond_count,
                        max_dist_count=max_dist_count
                    ))
                    prod_graph.append(get_feat(
                        token=prod[i],
                        vocab=self.vocabulary,
                        perm_idx=prod_aug[i][0],
                        suf_idx=prod_suf[i][0],
                        dual_task=0,
                        reaction_class=reaction_class,
                        max_deg=max_deg,
                        K=K,
                        kernel=kernel,
                        max_bond_count=max_bond_count,
                        max_dist_count=max_dist_count
                    ))
                featurize_result.append([prod, prod_graph, prod_aug, prod_suf, reac, reac_graph, reac_aug, reac_suf, reaction_class, reaction_len])

                if len(featurize_result) >= max_split_count:
                    split_record[f'featurize({mode})({self.vocab_name})-{split_count}.pkl'] = len(featurize_result)
                    self.save(
                        mode=mode,
                        featurize_result=featurize_result,
                        split_count=split_count,
                        length=len(featurize_result)
                    )
                    split_count += 1
                    featurize_result = []
            split_record[f'featurize({mode})({self.vocab_name})-{split_count}.pkl'] = len(featurize_result)
            self.save(
                mode=mode,
                featurize_result=featurize_result,
                split_count=split_count,
                length=len(featurize_result)
            )
            # np.savez(os.path.join(self.featurize_path, f'featurize({mode})({self.vocab_name})-record.npz'),
            #          record=split_record)
            with open(os.path.join(self.featurize_path, f'featurize({mode})({self.vocab_name})-record.pkl'), 'wb') as f:
                pickle.dump(split_record, f)
        self.logger.info(f'{self.preprocess_step}--> finish.')
        self.logger.info('\n')
    
    def save(
        self,
        mode: str,
        featurize_result: dict,
        split_count: int,
        length: int
    ):
        featurize_result = np.array(featurize_result, dtype=object)
        # np.savez(os.path.join(self.featurize_path, f'featurize({mode})({self.vocab_name})-{split_count}.npz'),
        #          featurize_result=featurize_result)
        with open(os.path.join(self.featurize_path, f'featurize({mode})({self.vocab_name})-{split_count}.pkl'), 'wb') as f:
            pickle.dump(featurize_result, f)
        self.logger.info(f'{self.preprocess_step}--> featurize({mode})({self.vocab_name})-{split_count}.pkl save finish, length={length}.')


if __name__ == '__main__':
    feat = Featurize(
        dataset_name='uspto_50k',
        vocab_name='uspto_50k'
    )
    # smi = '[H]C1=NC([H])=C([H])N1[H]'
    # graph = Chem.MolFromSmiles(smi)
    # smi_reverse = Chem.MolToSmiles(graph, rootedAtAtom=2)
    # a = get_feat(
    #     token=smi,
    #     vocab=feat.vocabulary
    # )
    # b = get_feat(
    #     token=smi_reverse,
    #     vocab=feat.vocabulary
    # )
    # pass
    # smi = 'C1CC1'
    # atom_feat, bond_idx, bond_feat = get_1hop_feat(
    #     token=smi,
    #     rvocab={id:token for token, id in feat.vocabulary.items()}
    # )
    # get_khop_feat(
    #     node_num=atom_feat.shape[0],
    #     bond_idx=bond_idx,
    #     bond_feat=bond_feat
    # )