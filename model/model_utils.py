import os
import re
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from typing import Optional
from torch_scatter import scatter
from copy import deepcopy as dcopy
from collections import deque
from torch.optim.lr_scheduler import _LRScheduler
from .preprocess.data_utils import canonicalize_smiles

CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpt')
PATTERN = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
REGEX = re.compile(PATTERN)

def setseed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def smi2tokenlist(smi: str):
    tokens = [token for token in REGEX.findall(smi)]
    return tokens

def smiles_augment(
    seq: list[list[int]],
    vocab: dict[str:int],
    rvocab: dict[int:str], # only includes smiles token
    augment_rate: Optional[float]=0.
):
    if augment_rate > 0:
        for i in range(len(seq)):
            if np.random.rand() < augment_rate:
                smi = [rvocab.get(id) for id in seq[i]]
                smi = ''.join(smi).split('.')
                smi_list = []
                for s in smi: # shuffle
                    sgraph = Chem.MolFromSmiles(s)
                    permute_smi = np.random.choice([atom.GetIdx() for atom in sgraph.GetAtoms()])
                    permute_smi = Chem.MolToSmiles(sgraph, rootedAtAtom=int(permute_smi))
                    smi_list.append(permute_smi)
                if len(smi_list) > 1 and np.random.rand() < 0.5: # reactant permute
                    smi_list = smi_list[::-1]
                smi_list = '.'.join(smi_list)
                smi_list = smi2tokenlist(smi_list)
                seq[i] = [vocab.get(token) for token in smi_list]
    return seq

def smiles_shuffle(
    seq: list[list[int]], # size(N, seq_len), not include <EOS>, <BOS> or other extra tokens
    vocab_id: list[int], # only includes smiles token
    gt_prob: Optional[float]=0.85,
    del_rate: Optional[float]=0.15,
    add_rate: Optional[float]=0.15,
    replace_rate: Optional[float]=0.15,
    shuffle_rate: Optional[float]=0.15,
    sample_weight: Optional[list[float]]=None, # the frequency of each token in dataset
):  
    shuffle_idx = np.random.rand(len(seq))
    if (shuffle_idx > gt_prob).any():
        shuffle_idx = np.where(shuffle_idx > gt_prob)[0]
        edit_sample = np.random.choice([_ for _ in range(4)], shuffle_idx.shape[0], replace=True)
        seq = np.array(seq, dtype=object)
        for idx, edit_id in zip(shuffle_idx, edit_sample):
            if int(len(seq[idx]) * del_rate) < 1: continue
            shuffle_seq = np.array(seq[idx], dtype=np.int16)
            seq_len = shuffle_seq.shape[-1]
            if edit_id == 0: # del
                del_idx = np.random.randint(0, seq_len, size=int(seq_len * del_rate + 0.5))
                del_idx = np.isin(np.arange(seq_len), del_idx, invert=True)
                shuffle_seq = shuffle_seq[del_idx]
            elif edit_id == 1: # add
                add_idx = np.random.randint(0, seq_len, size=int(seq_len * add_rate + 0.5))
                new_token = np.random.choice(vocab_id, add_idx.shape[0], replace=True, p=sample_weight)
                shuffle_seq = np.insert(shuffle_seq, add_idx, new_token, axis=0)
            elif edit_id == 2: # replace
                rep_idx = np.random.randint(0, seq_len, size=int(seq_len * replace_rate + 0.5))
                new_token = np.random.choice(vocab_id, rep_idx.shape[0], replace=True, p=sample_weight)
                shuffle_seq[rep_idx] = new_token
            elif edit_id == 3: # shuffle
                suf_idx = np.random.randint(0, seq_len, size=int(seq_len * shuffle_rate + 0.5))
                suf_idx = np.isin(np.arange(seq_len), suf_idx)
                shuffle_cache = shuffle_seq[suf_idx]
                shuffle_cache = np.random.permutation(shuffle_cache)
                shuffle_seq[suf_idx] = shuffle_cache
            
            seq[idx] = list(shuffle_seq)
        seq = list(seq)
    return seq

# param init for nn.Embedding
def embedding_init(embedding: nn.Embedding):
    fan_out = embedding.weight.size(1)
    std = 1.0 * math.sqrt(1.0 / float(fan_out))
    nn.init.normal_(embedding.weight, 0., std)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].fill_(0)
    return embedding

def linear_init(
    linear: nn.Linear,
    distribution: Optional[str]='normal',
    zero_bias: Optional[bool]=False,
    gain: Optional[float]=1.0
):
    if distribution == 'normal':
        nn.init.xavier_normal_(linear.weight, gain=gain)
    elif distribution == 'uniform':
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        if zero_bias:
            nn.init.zeros_(linear.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)
    return linear

def bias_init(bias: nn.Parameter):
    for param in bias:
        if param.dim() == 1:
            nn.init.normal_(param, 0., 1 / math.sqrt(param.size(-1)))
    return bias

def node_padding(
    node: torch.Tensor,
    node_num: torch.Tensor,
    max_len: Optional[int]=None
):
    if max_len == None:
        max_len = node_num.max()
    node_out = []
    split_node = torch.split(node, node_num.tolist(), dim=0)
    for len, graph in zip(node_num, split_node):
        operate = nn.ZeroPad2d((0, 0, 0, max_len - len))
        node_out.append(operate(graph))
    node_out = torch.stack(node_out, dim=0)
    return node_out

# generate softmax mask
def get_mask(
    q_len: int,
    k_len: int,
    batch_size: int,
    length: Optional[torch.Tensor]=None,
    attention_idx: Optional[list[torch.Tensor]]=None, # the value should take part in attention
    ignore_len: Optional[int]=0,
    causal: Optional[bool]=False,
    device: Optional[str]='cpu'
) -> torch.Tensor: 
    if length is None:
        if attention_idx is None:
            mask = torch.ones((batch_size, q_len, k_len), dtype=torch.bool, device=device)
        else:
            mask = torch.zeros((batch_size, k_len), dtype=torch.bool, device=device)
            mask[attention_idx] = True
            mask = mask.unsqueeze(1).repeat(1, q_len, 1)
            if causal:
                mask = torch.tril(mask, diagonal=0)
    else:
        length = length.unsqueeze(-1).repeat_interleave(q_len, -1)
        if causal:# for autoregressive mask
            mask = torch.tril(torch.ones((length.size(1), k_len), 
                              dtype=torch.int, device=device), 
                              diagonal=0).unsqueeze(0).repeat(batch_size, 1, 1)
            if ignore_len > 0:
                mask[:, :ignore_len, :ignore_len] = True
        else:
            mask = torch.zeros_like(length, device=device)\
                .unsqueeze(-1).repeat(1, 1, k_len)
            mask_idx = torch.arange((k_len), device=device)\
                .unsqueeze(0).repeat(length.size(1), 1)
            mask_idx = mask_idx.unsqueeze(0) < length.unsqueeze(-1)
            mask[mask_idx] = 1
    return mask.unsqueeze(1).to(torch.bool)

def search_result_process(
    tgt_seq: torch.Tensor,
    vocab: dict[str:int],
    rvocab: dict[int:str],
    beam_result: torch.Tensor
):
    '''canonicalize the predict smiles, then calculate top-k accuracy'''
    eos_ids, pad_ids = vocab['<EOS>'], vocab['<PAD>']
    batch_size, topk, res_len = beam_result.size()
    topk_acc = np.zeros((batch_size, topk))
    topk_invalid = np.zeros((batch_size, topk))
    tgt_seq, beam_result = tgt_seq.detach().cpu().numpy(),\
        beam_result.detach().cpu().numpy()
    all_smi = []
    for batch_id, batch_res in enumerate(beam_result):
        beam_smi = []
        for beam_id, beam_res in enumerate(batch_res):
            tgt = tgt_seq[batch_id]
            res = beam_res
            if (tgt == eos_ids).sum() > 0:
                tgt_eos = np.where(tgt == eos_ids)[0][0]
                tgt = tgt[:tgt_eos]
            if (res == eos_ids).sum() > 0:
                res_eos = np.where(res == eos_ids)[0][0]
                res = res[:res_eos]
            # tgt, res = tgt[((tgt != eos_ids) & (tgt != pad_ids))],\
            #     beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
            tgt_smi, res_smi = [rvocab[idx]for idx in tgt],\
                [rvocab[idx] for idx in res]
            tgt_smi, res_smi = ''.join(tgt_smi), ''.join(res_smi)
            if tgt_smi == 'CC':
                break  # problematic SMILES
            res_smi, valid = canonicalize_smiles(
                smi=res_smi,
                retain_smi=True,
                map_clear=False,
                cano_with_heavy_atom=False
            )
            beam_smi.append(res_smi)
            if not valid:
                topk_invalid[batch_id, beam_id] = 1
            else:
                # each batch only has one correct result
                if (res_smi == tgt_smi) and (topk_acc[batch_id].sum() == 0):
                    topk_acc[batch_id, beam_id] = 1
        beam_smi = ','.join(beam_smi)
        all_smi.append(f'{beam_smi}\n')
    return topk_acc.sum(axis=0), topk_invalid.sum(axis=0), all_smi


class Normalization(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: Optional[float]=1e-5,
        norm_type: Optional[str]='rmsnorm'
    ):
        super(Normalization, self).__init__()
        self.norm_type = norm_type
        self.norm = None
        self.weight = None

        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(d_model, eps)
        elif norm_type == 'rmsnorm':
            self.weight = nn.Parameter(torch.rand(d_model), requires_grad=True)
            self.eps = eps
        
        self.init_param()

    def init_param(self):
        if self.norm is not None:
            self.norm.reset_parameters()
        elif self.weight is not None:
            nn.init.ones_(self.weight)
    
    def forward(self, x:torch.Tensor):
        if self.norm_type == 'rmsnorm':
            mean_square = torch.square(x).mean(dim=-1, keepdim=True)
            x_out = x * torch.rsqrt(mean_square + self.eps) * self.weight
        else:
            x_out = self.norm(x)
        return x_out
    

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_out: Optional[int]=0,
        dropout: Optional[float]=0.1,
        init_gain: Optional[float]=1.,
        ffn_tpye: Optional[str]='glu'
    ):
        super(FeedForward, self).__init__()
        d_out = d_out if d_out > 0 else d_model
        self.ffn_type = ffn_tpye

        self.activate = nn.ReLU()
        self.drouput = nn.Dropout(dropout)
        self.Wout = nn.Linear(d_ff, d_out)
        if ffn_tpye == 'vanilla':
            self.Win = nn.Linear(d_model, d_ff)
        elif ffn_tpye == 'glu':
            self.Win = nn.Linear(d_model, d_ff * 2)
        
        self.init_param(gain=init_gain)
    
    def init_param(self, gain=1.):
        if self.ffn_type == 'vanilla':
            self.Win = linear_init(self.Win, gain=gain)
            self.Wout = linear_init(self.Wout, gain=gain)
        elif self.ffn_type == 'glu':
            self.Win = linear_init(self.Win)
            self.Wout = linear_init(self.Wout)
    
    def forward(self, x:torch.Tensor):
        x_in = self.Win(x)
        if self.ffn_type == 'vanilla':
            x_in = self.activate(x_in)
            x_in = self.drouput(x_in)
            x_out = self.Wout(x_in)
        elif self.ffn_type == 'glu':
            u, v = torch.chunk(x_in, 2, dim=-1)
            u = self.activate(u)
            x_out = self.drouput(u * v)
            x_out = self.Wout(x_out)
        return x_out


class AbsPositionalEmbedding(nn.Module):
    '''the base class for positional encoding'''
    def __init__(
        self,
        d_model: int,
        max_len: Optional[int]=2048,
        dropout: Optional[float]=0.
    ):
        super(AbsPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros((1, max_len, d_model))
        emb = torch.arange(max_len, dtype=torch.float).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe[:, :, 0::2] = torch.sin(emb)
        pe[:, :, 1::2] = torch.cos(emb)
        self.register_buffer('pe', pe)

    def forward(
        self, 
        x: torch.Tensor, 
        step=0
    ):
        x = x + self.pe[:, step:(x.size(1) + step), :]
        return x


class CrossEntropy(nn.Module):
    def __init__(
        self,
        vocab: dict[str:int],
        gamma: Optional[float]=0.,
        ignore_idx: Optional[int]=-1
    ):
        super(CrossEntropy, self).__init__()
        self.vocab = vocab
        self.gamma = gamma
        self.ignore_idx = ignore_idx

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        smooth: Optional[float]=0.1
    ):
        input_logsoftmax = input.log_softmax(dim=1)
        loss = F.nll_loss(input_logsoftmax, target, ignore_index=self.ignore_idx, reduction='none')
        
        if smooth > 0.:
            t_prob, f_prob = 1.0 - smooth, smooth
            f_idx = torch.arange(len(self.vocab), dtype=torch.long,
                                 device=input.device)[None, None, :].ne(target[:, :, None])
            ignore_idx = target.ne(self.ignore_idx)
            loss = loss * t_prob + (-input_logsoftmax.transpose(1, 2)[f_idx]).\
                reshape(loss.size(0), loss.size(1), -1).mean(dim=-1) * f_prob
            loss = loss * ignore_idx
        
        if self.gamma > 0.:
            occur_token = target.reshape(-1)
            occur_token_idx = [[_ for _ in range(target.size(0) * target.size(1))], occur_token.tolist()]
            occur_token_prob = input.softmax(dim=1).transpose(1, 2).\
                reshape(-1, len(self.vocab))
            occur_token_prob = occur_token_prob[occur_token_idx].\
                reshape(target.size(0), target.size(1))
            loss = loss * ((1 - occur_token_prob) ** self.gamma)
        
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        t: Optional[float]=0.2,
        eps: Optional[float]=1e-5
    ):
        super(ContrastiveLoss, self).__init__()
        self.t = t
        self.eps = eps
    
    def forward(
        self,
        input: torch.Tensor, # size(bsz, d_model) or size(len, d_model)
        label: Optional[torch.Tensor]=None, # size(bsz), same label means the positive pair, or length for each batch
        target: Optional[torch.Tensor]=None, # size(len, d_model)
        idx: Optional[list[torch.Tensor]]=None
    ):
        if target is None:
            batch_size = label.size(0)
            sim_matrix = F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2, eps=self.eps)
            sim_mask = torch.ones_like(sim_matrix) * (label.expand((batch_size, batch_size)).eq(label.expand((batch_size, batch_size)).t()))
            # unsim_mask = torch.ones_like(sim_matrix) - sim_mask

            sim_matrix = torch.exp(sim_matrix / self.t)
            matrix_diag = torch.diag(sim_matrix).diag_embed()
            sim_matrix = sim_matrix - matrix_diag
            sim = sim_mask * sim_matrix
            unsim = (sim_matrix - sim).sum(dim=-1, keepdim=True)
            loss = torch.div(sim, sim + unsim)
            loss[loss.le(0)] = 1
            loss = (-torch.log(loss)).sum()
            mask_diag = torch.diag(sim_mask).diag_embed()
            loss = loss / len(torch.nonzero(sim_mask - mask_diag))
        else:
            sim_dot = input.index_select(0, idx[0]) * target.index_select(0, idx[1])
            sim_norm_dot = torch.norm(input, p=2, dim=1, keepdim=True).index_select(0, idx[0]) *\
                torch.norm(target, p=2, dim=1, keepdim=True).index_select(0, idx[1])
            sim_norm_dot[sim_norm_dot < self.eps] = self.eps
            sim_matrix = torch.sum(torch.div(sim_dot, sim_norm_dot), dim=-1)
            sim_matrix = torch.exp(sim_matrix / self.t)

            # sim_matrix = F.cosine_similarity(input.unsqueeze(1), target.unsqueeze(0), dim=2, eps=self.eps)
            # sim_matrix = (input.unsqueeze(1) * target.unsqueeze(0)).sum(dim=-1)
            # sim_matrix = torch.exp(sim_matrix / self.t)
            loss = torch.div(sim_matrix[idx[0] == idx[1]], scatter(sim_matrix, idx[0], 0, dim_size=input.size(0), reduce='sum'))
            loss = (-torch.log(loss)).mean()
        return loss


class RsqrtLearningRate(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup: Optional[int]=8000,
        lr_factor: Optional[float]=1.0,
        max_lr: Optional[float]=1e-3
    ):
        self.d_model = d_model
        self.warmup = warmup
        self.basic_lr = [lr_factor]
        self.max_lr = max_lr
        super(RsqrtLearningRate, self).__init__(optimizer)
    
    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.d_model ** (-0.5) * min(step **(-0.5), step * self.warmup ** (-1.5))

        return [min(base_lr * scale, self.max_lr) for base_lr in self.basic_lr]


class CosineLearningRate(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: Optional[int]=8000,
        max_lr: Optional[float]=3e-4,
        min_lr: Optional[float]=1e-5,
        end_step: Optional[float]=150000
    ):
        self.warmup = warmup
        self.basic_lr = [max_lr]
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.end_step = end_step
        super(CosineLearningRate, self).__init__(optimizer)
    
    def get_lr(self):
        step = max(1, self._step_count)
        if step <= self.warmup:
            scale = step / self.warmup
            return [min(base_lr * scale, self.max_lr) for base_lr in self.basic_lr]
        else:
            scale = (self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1.0 + math.cos(((step - self.warmup) / (max(self.end_step, step) - self.warmup)) * math.pi))) / self.max_lr
            if scale * self.max_lr < self.min_lr:
                scale = self.min_lr / self.max_lr
            return [min(base_lr * scale, self.max_lr) for base_lr in self.basic_lr]


class Attention(nn.Module):
    def __init__(
        self,
        d_head: int,
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
    ):
        super(Attention, self).__init__()
        self.d_head = d_head
        self.pe_type = pe_type
        self.scale = 1 / math.sqrt(d_head)

        if self.pe_type == 'rope':
            # (1, max_len, d_model), [..., 0::2] = sin, [..., 1::2] = cos
            assert sin_emb.dim() == 3
            self.register_buffer('pe', sin_emb)
        else:
            self.pe = None
    
    def attention_scale(self, x:torch.Tensor):
        return x * self.scale

    def compute_rope(
        self,
        x: list[torch.Tensor],
        step: list[int]
    ):
        if self.pe_type != 'rope':
            return x
        else:
            assert len(x) == len(step)
            x_lens = [_.size(1) for _ in x]
            sin_pe = self.pe[..., 0::2].repeat_interleave(2, -1)
            cos_pe = self.pe[..., 1::2].repeat_interleave(2, -1)
            out = []
            for i in range(len(x)):
                x_in, x_len, t = x[i], x_lens[i], step[i]
                x_in_shift = torch.stack([-x_in[..., 1::2], x_in[..., 0::2]], dim=-1)
                x_in_shift = x_in_shift.reshape(x_in.shape)
                out.append(x_in * cos_pe[:, t:x_len + t, :] +
                           x_in_shift * sin_pe[:, t:x_len + t, :])
            return out


class ModelSave():
    """
    The model save class for checkpoint save and auto averaging.\\
    Args:
        strategy(str): mean/last, mean for top10 evaling accuracy \sum_{i\in\{1,3,5,10\}}Acc_i, last for the last training model
        save_num(int): the maximum number of checkpoint files
        q_len(int): the save queue length for checkpoint
        avg_tgt(int): the top-i model in queue for weights averaging
        const_save(list[int]): save the step consistently
        w1(float): the weight for top1 accuracy
    """
    def __init__(
        self,
        ckpt_dir: str,
        # if mean, model will save according to its mean(top1 + top3 + top5 + top10),
        # if last, it will save the last model
        strategy='mean',
        save_num=10,  # the maximum ckpt number
        q_len=10,
        avg_tgt=5,
        const_save=[],  # save these step consistently.
        w1=0.9
    ):
        self.ckpt_dir = ckpt_dir
        self.strategy = strategy
        self.q_len = q_len
        self.avg_tgt = avg_tgt
        self.const_save = const_save
        self.w1 = w1

        self.avg_list = []
        self.save_count = 0
        self.queue = deque([], maxlen=save_num) # [step, wacc]
        self.queue_dir = os.path.join(self.ckpt_dir, 'ckpt_queue.txt')
    
    def weight_acc(
        self,
        acc: list[float],
        w1: Optional[float]=None
    ) -> float:
        if w1 is None: w1 = self.w1
        return acc[0] * w1 + sum(acc[1:]) * (1 - w1) / max(len(acc[1:]), 1)
    
    def remove_ckpt(self, ckpt: list):
        step = ckpt[0]
        save_path = os.path.join(self.ckpt_dir, f'{step}.pt')
        if step not in self.const_save:
            if os.path.exists(save_path): os.remove(save_path)
    
    def save_ckpt(
        self,
        step: int,
        model: nn.Module
    ):
        save_path = os.path.join(self.ckpt_dir, f'{step}.pt')
        torch.save({'model': dcopy(model.state_dict())}, save_path)
    
    def save(
        self,
        model: nn.Module,
        step: int,
        acc: list[float]
    ):
        wacc = acc[0] * self.w1 + sum(acc[1:]) * (1 - self.w1) / max(len(acc[1:]), 1)
        if len(self.queue) == 0:
            self.queue.append([step, wacc])
            self.save_ckpt(step, model)
        else:
            if self.strategy == 'last':
                if len(self.queue) == self.queue.maxlen:
                    pop_ckpt = self.queue.popleft()
                    self.remove_ckpt(pop_ckpt)
                self.queue.append([step, wacc])
                self.save_ckpt(step, model)
            else:
                for i in range(len(self.queue) - 1, -1, -1):
                    ckpt = self.queue[i]
                    ckpt_wacc = ckpt[1]
                    if wacc <= ckpt_wacc:
                        if i == 0 and len(self.queue) < self.queue.maxlen:
                            self.queue.insert(i, [step, wacc])
                            self.save_ckpt(step, model)
                        else:
                            continue
                    if wacc > ckpt_wacc:
                        if len(self.queue) == self.queue.maxlen:
                            pop_ckpt =self.queue.popleft()
                            self.remove_ckpt(pop_ckpt)
                            self.queue.insert(i, [step, wacc])
                            self.save_ckpt(step, model)
                            break
                        else:
                            self.queue.insert(i + 1, [step, wacc])
                            self.save_ckpt(step, model)
                            break
        if step in self.const_save and step not in [elem[0] for elem in self.queue]:
            self.save_ckpt(step, model)
        self.save_count += 1
        if self.save_count % self.q_len == 0:
            avg_list = [self.queue[i][0] for i in range(len(self.queue) - 1, len(self.queue) - self.avg_tgt - 1, -1)]
            self.avg_list.append(','.join([str(i) for i in avg_list]))
            self.model_average(avg_list, 'AVG' + str(len(self.avg_list)))
        with open(self.queue_dir, mode='w') as f:
            for i in range(len(self.queue) - 1, -1, -1):
                ckpt_step, ckpt_wacc = self.queue[i]
                f.writelines('{i}: {ckpt_step} wacc ---> {ckpt_wacc:.6}\n'.format(
                    i=i + 1, ckpt_step=ckpt_step, ckpt_wacc=ckpt_wacc
                ))
            if len(self.avg_list) > 0:
                for i in range(len(self.avg_list)):
                    f.writelines('avg {i} source models: {avglist}\n'.format(
                        i=i + 1, avglist=self.avg_list[i]
                    ))
    
    def model_average(
        self,
        step_list: list[int],
        model_name: Optional[str]='AVG_MAIN'
    ):
        for i, step in enumerate(step_list):
            load_path = os.path.join(self.ckpt_dir, f'{step}.pt')
            model = torch.load(load_path, map_location='cpu')
            model_param = model['model']
            if i == 0:
                avg_param = model_param
            else:
                for (k, v) in avg_param.items():
                    avg_param[k].mul_(i).add_(model_param[k]).div_(i + 1)
        save_path = os.path.join(self.ckpt_dir, model_name + '.pt')
        torch.save({'model':avg_param}, save_path)
    
    def load(
        self,
        model_name: str | int,
        model: nn.Module,
        device: Optional[str]='cpu'
    ):
        load_path = os.path.join(self.ckpt_dir, f'{model_name}.pt')
        model_param = torch.load(load_path, map_location=device)
        model.load_state_dict(model_param['model'])
        return model
    
    def eval_record(
        self,
        mode: str,
        model_name: str | int,
        seq_acc: list[float],
        seq_invalid: list[float],
        beam_size: int,
        T: float
    ):
        write_path = os.path.join(self.ckpt_dir, f'{model_name} {mode} {beam_size} {T:.3}.txt')
        with open(write_path, 'w') as f:
            for k, (acc, invalid) in enumerate(zip(seq_acc, seq_invalid)):
                f.writelines('top{k}---> acc={acc:.6}%, invalid={invalid:.6}%\n'.format(
                    k=k + 1, acc=acc * 100, invalid=invalid * 100
                ))
            f.writelines('weight average accuracy={acc:.6}'.format(
                acc=seq_acc[0] * self.w1 + sum(seq_acc[1:]) * (1 - self.w1) / max(len(seq_acc[1:]), 1)
            ))


class MixUp(nn.Module):
    def __init__(
        self,
        mixup_process: Optional[bool]=True,
        alpha: Optional[float]=1.0,
        task: Optional[str]='dualtask',
        device: Optional[str]='cuda:0'
    ):
        super(MixUp, self).__init__()
        self.mixup_process = mixup_process
        self.task = task
        self.lam_sample = torch.distributions.beta.Beta(alpha, alpha)
        self.lam = None
        self.perm_idx = None
        self.device = device
    
    def sample(self, batch_size: int, num_list: Optional[list[int]]=None):
        if self.mixup_process:
            if self.task == 'dualtask':
                lam1, lam2 = self.lam_sample.sample((num_list[0],)).to(self.device),\
                    self.lam_sample.sample((num_list[1],)).to(self.device)
                perm_idx1, perm_idx2 = torch.randperm(num_list[0], device=self.device),\
                    torch.randperm(num_list[1], device=self.device) + num_list[0]
                self.lam = torch.cat([lam1, lam2], 0)
                self.perm_idx = torch.cat([perm_idx1, perm_idx2], 0)
            else:
                self.lam = self.lam_sample.sample((batch_size,)).to(self.device)
                self.perm_idx = torch.randperm(batch_size, device=self.device)
            self.lam[self.lam < 0.5] = 1.0 - self.lam[self.lam < 0.5]
        else:
            self.lam = None
            self.perm_idx = None
    
    def forward(self, x: torch.Tensor, x_len: Optional[torch.Tensor]=None):
        if self.mixup_process:
            if x.dim() == 3:
                lam = self.lam.reshape(-1, 1, 1) # size(bsz, l, h)
            elif x.dim() == 2:
                lam = self.lam.reshape(-1, 1)# size(bsz, h)
            if x_len is not None:
                x_len, _ = torch.max(torch.stack([x_len, x_len.index_select(0, self.perm_idx)], dim=1), dim=-1)
            return x * lam + (1.0 - lam) * x.index_select(0, self.perm_idx), x_len
        else:
            return x, x_len
        
    
