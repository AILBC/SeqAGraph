import os
import re
import pickle
import torch
import numpy as np

from typing import Optional
from copy import deepcopy as dcopy
from torch.utils.data import IterableDataset, DataLoader
from .preprocess.dataset_base import DATA_DIR
from .preprocess.chem_utils import BOND_FDIM


class ReactionData():
    def __init__(
        self,
        seq: Optional[torch.Tensor]=None,
        seq_len: Optional[torch.Tensor]=None,
        seq_aug: Optional[torch.Tensor]=None,
        seq_suf: Optional[torch.Tensor]=None,
        tgt: Optional[torch.Tensor]=torch.tensor([]),
        tgt_len: Optional[torch.Tensor]=torch.tensor([]),
        tgt_aug: Optional[torch.Tensor]=None,
        tgt_suf: Optional[torch.Tensor]=None,
        graph_len: Optional[torch.Tensor]=None,
        f_atom: Optional[torch.Tensor]=None,
        bond_idx: Optional[list[torch.Tensor]]=None,
        bond_neibor: Optional[list[torch.Tensor]]=None,
        f_bond: Optional[list[torch.Tensor]]=None,
        deg: Optional[torch.Tensor]=None,
        dist: Optional[torch.Tensor]=None,
        query_idx: Optional[torch.Tensor]=None,
        key_idx: Optional[torch.Tensor]=None,
        batch_bond_idx: Optional[list[torch.Tensor]]=None,
        align_idx: Optional[list[torch.Tensor]]=None,
        f_class: Optional[torch.Tensor]=None,
        f_task: Optional[torch.Tensor]=None,
        task: Optional[str]=None
    ):
        self.seq = seq
        self.seq_len = seq_len
        self.seq_aug = seq_aug
        self.seq_suf = seq_suf
        self.tgt = tgt
        self.tgt_len = tgt_len
        self.tgt_aug = tgt_aug
        self.tgt_suf = tgt_suf
        self.graph_len = graph_len
        self.f_atom = f_atom
        self.bond_idx = bond_idx
        self.bond_neibor = bond_neibor
        self.f_bond = f_bond
        self.deg = deg
        self.dist = dist
        self.query_idx = query_idx
        self.key_idx = key_idx
        self.batch_bond_idx = batch_bond_idx
        self.align_idx = align_idx
        self.f_class = f_class
        self.f_task = f_task
        self.task = task
    
    def to(
        self,
        device
    ):
        self.seq = self.seq.to(device)
        self.seq_len = self.seq_len.to(device)
        self.seq_aug = self.seq_aug.to(device)
        self.seq_suf = self.seq_suf.to(device)
        self.tgt = self.tgt.to(device)
        self.tgt_len = self.tgt_len.to(device)
        self.tgt_aug = self.tgt_aug.to(device)
        self.tgt_suf = self.tgt_suf.to(device)
        self.graph_len = self.graph_len.to(device)
        self.f_atom = self.f_atom.to(device)
        self.bond_idx = [_.to(device) for _ in self.bond_idx]
        self.bond_neibor = [_.to(device) for _ in self.bond_neibor]
        self.f_bond = [_.to(device) for _ in self.f_bond]
        self.deg = self.deg.to(device)
        self.dist = self.dist.to(device)
        self.query_idx = self.query_idx.to(device)
        self.key_idx = self.key_idx.to(device)
        self.batch_bond_idx = [_.to(device) for _ in self.batch_bond_idx]
        self.align_idx = [_.to(device) for _ in self.align_idx]
        self.f_class = self.f_class.to(device)
        self.f_task = self.f_task.to(device)
        return self
    
    def pin_memory(self):
        self.seq = self.seq.pin_memory()
        self.seq_len = self.seq_len.pin_memory()
        self.seq_aug = self.seq_aug.pin_memory()
        self.seq_suf = self.seq_suf.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.tgt_len = self.tgt_len.pin_memory()
        self.tgt_aug = self.tgt_aug.pin_memory()
        self.tgt_suf = self.tgt_suf.pin_memory()
        self.graph_len = self.graph_len.pin_memory()
        self.f_atom = self.f_atom.pin_memory()
        self.bond_idx = [_.pin_memory() for _ in self.bond_idx]
        self.bond_neibor = [_.pin_memory() for _ in self.bond_neibor]
        self.f_bond = [_.pin_memory() for _ in self.f_bond]
        self.deg = self.deg.pin_memory()
        self.dist = self.dist.pin_memory()
        self.query_idx = self.query_idx.pin_memory()
        self.key_idx = self.key_idx.pin_memory()
        self.batch_bond_idx = [_.pin_memory() for _ in self.batch_bond_idx]
        self.align_idx = [_.pin_memory() for _ in self.align_idx]
        self.f_class = self.f_class.pin_memory()
        self.f_task = self.f_task.pin_memory()
        return self
        

class ReactionDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        vocab_name: str,
        mode: str,
        task: str,
        reaction_class: bool,
        token_limit: int,
        K: int,
        kernel: str,
        dist_block: list[int],
        split_data_len: int,
        shuffle: str,
        augment_N: int,
        max_perm_idx: int,
        
    ):
        super(ReactionDataset, self).__init__()
        self.mode = mode
        self.name = dataset_name
        self.vocab_name = vocab_name
        self.task = task
        self.reaction_class = reaction_class
        self.token_limit = token_limit
        self.K = K
        self.dist_block = dist_block
        self.shuffle = shuffle
        self.augment_N = augment_N
        self.max_perm_idx = max_perm_idx
        self.data_dir = os.path.join(DATA_DIR, self.name)
        self.process_dir = os.path.join(self.data_dir, 'process')
        self.process_dir = os.path.join(self.process_dir, f'K_{K}({kernel})')

        if split_data_len <= 0:
            with open(os.path.join(self.process_dir, f'featurize({mode})({self.vocab_name})-record.pkl'), 'rb') as f:
                load_path = pickle.load(f)
            self.load_path, self.load_length = list(load_path.keys()), list(load_path.values())
        else:
            self.load_path, self.load_length = [f'featurize({self.mode})({self.vocab_name})-s{split_data_len}.pkl'], [split_data_len]
        self.load_path = np.array(self.load_path, dtype=object)
        self.load_length = np.array(self.load_length, dtype=np.int32)
        self.data_cache = []
        
        self.load_vocab()

    def file_shuffle(
        self
    ):
        self.shuffle_idx = np.random.permutation(self.data_cache.shape[0])

    def load_data(self):
        for i, (file_name, file_length) in enumerate(zip(self.load_path, self.load_length)):
            with open(os.path.join(self.process_dir, file_name), 'rb') as f:
                self.data_cache.append(pickle.load(f))
            for data in self.data_cache[i]:
                yield data

    def load_batch(self):
        if len(self.data_cache) < len(self.load_length): # load file check
            for i in range(len(self.data_cache), len(self.load_length), 1):
                with open(os.path.join(self.process_dir, self.load_path[i]), 'rb') as f:
                    self.data_cache.append(pickle.load(f))
        
        if isinstance(self.data_cache, list):
            self.data_cache = np.concatenate(self.data_cache, axis=0)

        if self.shuffle:
            self.file_shuffle()
            for i in self.shuffle_idx:
                yield self.data_cache[i]
        else:
            for i in range(self.data_cache.shape[0]):
                    yield self.data_cache[i]


    def process_data(
        self,
        data: list,
    ) -> ReactionData:
        data_len = [_[-1] for _ in data]
        if self.token_limit > 0 and self.mode == 'train':
            if max(data_len) * len(data_len) > self.token_limit:
                for i in range(-1, -len(data_len), -1):
                    if max(data_len[:i]) * len(data_len[:i]) <= self.token_limit: break
                data = data[:i]
        data_dict = {'prod': [], 'prod_graph': [], 'prod_aug': [], 'prod_suf': [], 'reac': [], 'reac_graph': [], 'reac_aug': [], 'reac_suf': [], 'reaction_class': []}
        for d in data:
            if self.augment_N > 1:
                i = 0 if np.random.rand() < 0.5 else np.random.choice(self.augment_N - 1, 1)[0] + 1
            else:
                i = 0
            data_dict['prod'].extend([d[0][i]])
            data_dict['prod_graph'].extend([d[1][i]])
            data_dict['prod_aug'].extend([d[2][i]])
            data_dict['prod_suf'].extend([d[3][i]])
            data_dict['reac'].extend([d[4][i]])
            data_dict['reac_graph'].extend([d[5][i]])
            data_dict['reac_aug'].extend([d[6][i]])
            data_dict['reac_suf'].extend([d[7][i]])
            data_dict['reaction_class'].extend([d[8]])

        batch_data = ReactionData()
        prod_num, reac_num = len(data_dict['prod']), len(data_dict['reac'])

        def comput_length(
            batch_data: ReactionData,
            input_seq: list[list[int]],
            input_graph: list,
            tgt_seq: list[list[int]]
        ):
            input_seq_len, tgt_seq_len, graph_len = [], [], []
            for seq_i, seq_j, g in zip(input_seq, tgt_seq, input_graph):
                input_seq_len.append(len(seq_i) + 1)
                tgt_seq_len.append(len(seq_j) + 1) # EOS
                graph_len.append(g['atom_feat'].shape[0])
            seq_max_len = max(input_seq_len + tgt_seq_len)
            batch_data.seq_len = torch.tensor(input_seq_len, dtype=torch.long)
            batch_data.tgt_len = torch.tensor(tgt_seq_len, dtype=torch.long)
            batch_data.graph_len = torch.tensor(graph_len, dtype=torch.long)
            return batch_data, seq_max_len

        def process_sequence(
            batch_data: ReactionData,
            sequence: list[list[int]],
            max_length: list[int], #[input sequence, target sequence]
            eos: int,
            pad: int
        ) -> ReactionData:
            for id, (seq, max_len) in enumerate(zip(sequence, max_length)):
                for i in range(len(seq)):
                    seq[i] += [eos]
                    seq[i] += [pad] * (max_len - len(seq[i]))
                if id == 0:
                    batch_data.seq = torch.tensor(seq, dtype=torch.long)
                elif id == 1:
                    batch_data.tgt = torch.tensor(seq, dtype=torch.long)
            return batch_data

        def process_graph(
            batch_data: ReactionData, 
            graph: dict,
            graph_len: list[int],
            K: int,
            dist_block: list[int]
        ) -> ReactionData:
            batch_dist = []
            batch_deg = []
            batch_atom_feat = []
            batch_bond_idx = [[] for _ in range(K)]
            batch_bond_feat = [[] for _ in range(K)]
            batch_bond_neibor = [[] for _ in range(K)]
            query_idx, key_idx = [], []
            bond_batch_idx = [[] for _ in range(K)]
            batch_align_idx = [[], []]
            bias = 0
            bond_bias = [0 for _ in range(K)]

            graph_len_shift = np.array([0] + graph_len[:-1], dtype=np.int32)
            graph_len_cumsum = np.cumsum(graph_len_shift ** 2, axis=0)
            
            for i, (g, length) in enumerate(zip(graph, graph_len)):
                atom_feat, bond_idx, bond_feat, bond_neibor, deg, dist, align_idx = g['atom_feat'], g['bond_idx'], g['bond_feat'], g['bond_neibor'],\
                    g['deg'], g['dist'], g['align_idx']

                bond_num = [len(_) for _ in bond_feat]

                batch_dist = np.concatenate([batch_dist, dist.reshape(-1)], axis=-1) if\
                    isinstance(batch_dist, np.ndarray) else dist.reshape(-1)
                batch_deg = np.concatenate([batch_deg, deg], axis=-1) if\
                    isinstance(batch_deg, np.ndarray) else deg
                if len(atom_feat) <= 1: atom_feat = np.expand_dims(atom_feat, axis=0)
                batch_atom_feat = np.concatenate([batch_atom_feat, atom_feat], axis=0) if\
                    isinstance(batch_atom_feat, np.ndarray) else atom_feat
                bond_idx_range = torch.arange(bias, bias + length, dtype=torch.long)
                query_idx = torch.cat([query_idx, bond_idx_range.repeat_interleave(length, dim=0)], dim=-1) if\
                    isinstance(query_idx, torch.Tensor) else bond_idx_range.repeat_interleave(length, dim=0)
                key_idx = torch.cat([key_idx, bond_idx_range.repeat(length)], dim=-1) if\
                    isinstance(key_idx, torch.Tensor) else bond_idx_range.repeat(length)
                
                batch_align_idx[0] = np.concatenate([batch_align_idx[0], np.array([i] * len(align_idx), dtype=np.int16)], axis=0) if\
                    isinstance(batch_align_idx[0], np.ndarray) else np.array([i] * len(align_idx), dtype=np.int16)
                batch_align_idx[1] = np.concatenate([batch_align_idx[1], align_idx], axis=0) if\
                    isinstance(batch_align_idx[1], np.ndarray) else align_idx
                
                if bond_idx[0].shape[-1] >= 1:
                    for k in range(K):
                        bond_idx_k, bond_feat_k = bond_idx[k].astype(np.int32), bond_feat[k]
                        if bond_idx_k.shape[-1] >= 1:
                            bond_idx_k[0] += bias
                            bond_idx_k[1] += bias
                            batch_bond_idx[k] = np.concatenate([batch_bond_idx[k], bond_idx_k], axis=-1) if\
                                isinstance(batch_bond_idx[k], np.ndarray) else bond_idx_k
                            batch_bond_feat[k] = np.concatenate([batch_bond_feat[k], bond_feat_k], axis=0) if\
                                isinstance(batch_bond_feat[k], np.ndarray) else bond_feat_k
                            # calculate index in global attention matrix
                            bond_batch_idx[k].extend(((bond_idx_k[0] - bias) * length +
                                                      (bond_idx_k[1] - bias) + graph_len_cumsum[i]).tolist())
                    for k in range(1):
                        if bond_neibor[k].shape[-1] >= 1:
                            bond_neibor_k = bond_neibor[k].astype(np.int64)
                            bond_neibor_k[0] += bond_bias[0]
                            bond_neibor_k[1] += bond_bias[0]
                            batch_bond_neibor[k] = np.concatenate([batch_bond_neibor[k], bond_neibor_k], axis=-1) if\
                                isinstance(batch_bond_neibor[k], np.ndarray) else bond_neibor_k
                
                bias += length
                bond_bias = [i + j for i, j in zip(bond_bias, bond_num)]
            
            for id, blk in enumerate(dist_block):
                if isinstance(blk, list):
                    batch_dist[(batch_dist >= blk[0]) & (batch_dist < blk[1])] = id
                else:
                    batch_dist[batch_dist == blk] = id
            min_dist = dist_block[0][0] if isinstance(dist_block[0], list) else dist_block[0]
            max_dist = dist_block[-1][-1] if isinstance(dist_block[-1], list) else dist_block[-1]
            batch_dist[(batch_dist < min_dist) | (batch_dist > max_dist)] = len(dist_block)

            batch_data.dist = torch.tensor(batch_dist, dtype=torch.long)
            batch_data.deg = torch.tensor(batch_deg, dtype=torch.long)
            batch_data.f_atom = torch.tensor(batch_atom_feat, dtype=torch.long)
            batch_data.bond_idx = [torch.tensor(_, dtype=torch.long) for _ in batch_bond_idx]
            batch_data.bond_neibor = [torch.tensor(_, dtype=torch.long) for _ in batch_bond_neibor]
            batch_data.f_bond = [torch.tensor(_, dtype=torch.long) for _ in batch_bond_feat]
            batch_data.query_idx = query_idx
            batch_data.key_idx = key_idx
            batch_data.batch_bond_idx = [torch.tensor(_, dtype=torch.long) for _ in bond_batch_idx]
            batch_data.align_idx = [torch.tensor(_, dtype=torch.long) for _ in batch_align_idx]
            return batch_data

        batch_data.task = self.task
        if self.task == 'dualtask':
            input_graph, input_seq, target = dcopy(data_dict['prod_graph']) + dcopy(data_dict['reac_graph']), dcopy(data_dict['prod']) + dcopy(data_dict['reac']),\
                dcopy(data_dict['reac']) + dcopy(data_dict['prod'])
            input_aug, input_suf = dcopy(data_dict['prod_aug']) + dcopy(data_dict['reac_aug']), dcopy(data_dict['prod_suf']) + dcopy(data_dict['reac_suf'])
            tgt_aug, tgt_suf = dcopy(data_dict['reac_aug']) + dcopy(data_dict['prod_aug']), dcopy(data_dict['reac_suf']) + dcopy(data_dict['prod_suf'])
            batch_data.f_task = torch.tensor([0] * prod_num + [1] * reac_num, dtype=torch.long)
            if self.reaction_class:
                batch_data.f_class = torch.tensor(data_dict['reaction_class'] * 2, dtype=torch.long)
            else:
                batch_data.f_class = torch.full((prod_num + reac_num,), 0, dtype=torch.long)
        elif self.task == 'retrosynthesis':
            input_graph, input_seq, target = dcopy(data_dict['prod_graph']), dcopy(data_dict['prod']), dcopy(data_dict['reac'])
            input_aug, input_suf = dcopy(data_dict['prod_aug']), dcopy(data_dict['prod_suf'])
            tgt_aug, tgt_suf = dcopy(data_dict['reac_aug']), dcopy(data_dict['reac_suf'])
            batch_data.f_task = torch.tensor([0] * prod_num, dtype=torch.long)
            if self.reaction_class:
                batch_data.f_class = torch.tensor(data_dict['reaction_class'], dtype=torch.long)
            else:
                batch_data.f_class = torch.full((prod_num,), 0, dtype=torch.long)
        elif self.task == 'forwardsynthesis':
            input_graph, input_seq, target = dcopy(data_dict['reac_graph']), dcopy(data_dict['reac']), dcopy(data_dict['prod'])
            input_aug, input_suf = dcopy(data_dict['reac_aug']), dcopy(data_dict['reac_suf'])
            tgt_aug, tgt_suf = dcopy(data_dict['prod_aug']), dcopy(data_dict['prod_suf'])
            batch_data.f_task = torch.tensor([1] * reac_num, dtype=torch.long)
            if self.reaction_class:
                batch_data.f_class = torch.tensor(data_dict['reaction_class'], dtype=torch.long)
            else:
                batch_data.f_class = torch.full((reac_num,), 0, dtype=torch.long)

        batch_data.seq_aug = torch.tensor(input_aug, dtype=torch.long)
        batch_data.seq_suf = torch.tensor(input_suf, dtype=torch.long)
        batch_data.tgt_aug = torch.tensor(tgt_aug, dtype=torch.long)
        batch_data.tgt_suf = torch.tensor(tgt_suf, dtype=torch.long)
        batch_data, max_len = comput_length(
            batch_data=batch_data,
            input_seq=input_seq,
            input_graph=input_graph,
            tgt_seq=target
        )
        batch_data = process_sequence(
            batch_data=batch_data,
            sequence=[input_seq, target],
            max_length=[max_len, max_len],
            eos=self.vocabulary['<EOS>'],
            pad=self.vocabulary['<PAD>']
        )
        batch_data = process_graph(
            batch_data=batch_data,
            graph=input_graph,
            graph_len=batch_data.graph_len.tolist(),
            K=self.K,
            dist_block=self.dist_block,
        )
        return batch_data
            
    def load_vocab(
        self
    ):
        vocab_path = os.path.join(DATA_DIR, f'vocabulary({self.vocab_name}).txt')
        token, count = [], []
        with open(vocab_path, 'r') as f:
            for data in f:
                t, c = data.strip().split('\t')
                token.append(t)
                count.append(int(c))
        atom_pattern = re.compile(r'[A-Za-z]')
        extra_vocab = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        self.raw_vocabulary = {t:id for id, t in enumerate(token)}
        self.noatom_token_idx = []
        for (t, id) in self.raw_vocabulary.items():
            if not bool(re.search(atom_pattern, t)): self.noatom_token_idx.append(id)
        self.noatom_token_idx.extend([len(self.raw_vocabulary) + i for i in range(len(extra_vocab))])
        self.raw_reverse_vocabulary = {id:t for id, t in enumerate(token)}
        vocab_sum = sum(count)
        self.vocabulary_rate = [i / vocab_sum for i in count] 
        token.extend(extra_vocab)
        self.vocabulary = {t:id for id, t in enumerate(token)}


    def __iter__(self):
        return self.load_data() if len(self.data_cache) == 0 else self.load_batch()
    
    def __len__(self):
        return self.load_length.sum() 