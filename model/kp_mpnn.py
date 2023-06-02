import math
import torch
import torch.nn as nn

from typing import Optional
from torch_scatter import scatter
from .batch_loader import ReactionData
from .preprocess.chem_utils import NODE_FEAT_LIST, BOND_FEAT_LIST, BOND_TYPE_NUM
from .model_utils import embedding_init, linear_init

# NODE_FEAT_LIST = NODE_FEAT_LIST[:9] + NODE_FEAT_LIST[11:]

def get_onehot(
    label: torch.Tensor,
    num_class: int
) -> torch.Tensor:
    assert label.dim() == 1
    negative = torch.where(label == -1)
    label[negative] = 0
    one_hot = torch.zeros(label.size(0), num_class, dtype=torch.long, device=label.device)
    one_hot.scatter_(1, label.reshape(-1, 1), value=1)
    one_hot[negative] = 0
    return one_hot


class InitialEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        K: int,
        max_bond_count: int,
        max_dist_count: int,
        reaction_class=False
    ):
        super(InitialEmbedding, self).__init__()
        assert d_model % K == 0
        self.reaction_class = reaction_class
        self.d_emb = d_model

        self.node_emb = nn.Linear(sum(NODE_FEAT_LIST), d_model)
        self.bond_emb = nn.ModuleList()
        self.bond_emb.append(nn.Linear(sum(BOND_FEAT_LIST), d_model))
        if K > 1:
            for i in range(1, K, 1):
                self.bond_emb.append(nn.Embedding(max_bond_count + 1, d_model, padding_idx=0))

        self.init_param()
    
    def init_param(self):
        self.node_emb = linear_init(self.node_emb)
        for i in range(len(self.bond_emb)):
            if isinstance(self.bond_emb[i], nn.Embedding):
                self.bond_emb[i] = embedding_init(self.bond_emb[i])
            else:
                self.bond_emb[i] = linear_init(self.bond_emb[i])
    
    def get_onehot(
        self,
        data: ReactionData
    ) -> ReactionData:
        atom_emb = None
        f_atom = data.f_atom
        if not self.reaction_class: f_atom[..., -1] = -1
        else: f_atom[..., -1] -= 1
        # f_atom = torch.cat([f_atom[:, :9], f_atom[:, 11:]], dim=-1)
        for i, f_list in enumerate(NODE_FEAT_LIST):
            atom_emb = get_onehot(f_atom[..., i], f_list) if atom_emb is None\
                else torch.cat([atom_emb, get_onehot(f_atom[..., i], f_list)], dim=-1)
        data.f_atom = atom_emb.to(torch.float)

        bond_emb = None
        f_bond = data.f_bond[0]
        for i, f_list in enumerate(BOND_FEAT_LIST):
            bond_emb = get_onehot(f_bond[..., i], f_list) if bond_emb is None\
                else torch.cat([bond_emb, get_onehot(f_bond[..., i], f_list)], dim=-1)
        data.f_bond[0] = bond_emb.to(torch.float)
        return data
    
    def forward(
        self,
        data: ReactionData
    ) -> ReactionData:
        data = self.get_onehot(data)
        data.f_atom = self.node_emb(data.f_atom)
        for i in range(len(self.bond_emb)):
            data.f_bond[i] = self.bond_emb[i](data.f_bond[i])
        return data


class KhopCombine(nn.Module):
    """
    A method from 'How Powerful are K-hop Message Passing Graph Neural Networks' to combine the k-hop messages.
    """
    def __init__(
        self,
        d_model: int,
        d_score: int
    ):
        super(KhopCombine, self).__init__()
        self.gru = nn.GRU(d_model, d_score, 1, batch_first=True, bidirectional=True)

        self.init_param()
    
    def init_param(self):
        for param in self.gru.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param)
    
    def forward(
        self,
        x: torch.Tensor #size(N, K, d_model)
    ):
        self.gru.flatten_parameters()
        score, _ = self.gru(x)
        score = torch.softmax(score.sum(dim=-1), dim=1)
        return (x * score.unsqueeze(-1)).sum(dim=1) #size(N, d_model)


class SumCombine(nn.Module):
    """
    Method from GINE+
    """
    def __init__(
        self,
        K: int,
        d_model: Optional[int]=256
    ):
        super(SumCombine, self).__init__()
        self.K = K
        self.eps = torch.nn.Parameter(torch.randn((1, self.K)), requires_grad=True)

        self.init_param()

    def init_param(self):
        nn.init.zeros_(self.eps)
    
    def forward(self, x: torch.Tensor):
        return torch.sum((1 + self.eps.unsqueeze(-1)) * x, dim=1)


class DMPNN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: Optional[float]=0.
    ):
        super(DMPNN, self).__init__()
        self.d_node = d_model
        self.d_bond = d_model
        self.d_model = d_model
        
        self.dropout = nn.Dropout(dropout)
        self.mess_emb = nn.Sequential(nn.Linear(self.d_node + self.d_bond, d_model), nn.GELU())
        self.Wz = nn.Linear(self.d_node + self.d_bond + d_model, d_model)
        self.Wr = nn.Linear(self.d_node + self.d_bond + d_model, d_model)
        self.W = nn.Linear(self.d_node + self.d_bond, d_model)
        self.U = nn.Linear(d_model, d_model, bias=False)

        self.out = nn.Sequential(nn.Linear(self.d_node + d_model, d_model), nn.GELU())
        
        self.init_param()
    
    def init_param(self):
        self.mess_emb[0] = linear_init(self.mess_emb[0])
        self.Wz = linear_init(self.Wz)
        self.Wr = linear_init(self.Wr)
        self.W = linear_init(self.W)
        self.U = linear_init(self.U)
        self.out[0] = linear_init(self.out[0])
    
    def forward(
        self,
        f_mess: torch.Tensor,
        f_node: torch.Tensor, #size(N, d_node)
        bond_idx: torch.Tensor, #len(1)
        bond_neibor: torch.Tensor, #len(1)
        f_bond: torch.Tensor #len(1)
    ):
        src, tgt = bond_idx
        nei_src, nei = bond_neibor
        bond_num = src.size(0)
        h_ij = torch.cat([f_node.index_select(0, src), f_bond], dim=-1)
        h_ki = h_ij.index_select(0, nei_src)
        if f_mess is None:
            m_ij = self.mess_emb(h_ij)
        else:
            m_ij = f_mess.clone()

        m_ki = m_ij.index_select(0, nei)
        s_ij = scatter(m_ki, nei_src, 0, dim_size=bond_num, reduce='sum')
        z_ij = torch.sigmoid(self.Wz(torch.cat([h_ij, s_ij], dim=-1)))

        r_ki = torch.sigmoid(self.Wr(torch.cat([h_ki, m_ki], dim=-1)))
        r_ij = scatter(r_ki * m_ki, nei_src, 0, dim_size=bond_num, reduce='sum')
        m_ij = torch.tanh(self.W(h_ij) + self.U(r_ij))
        m_ij = (1 - z_ij) * s_ij + z_ij * m_ij
        m_ij = self.dropout(m_ij)
        
        m_j = scatter(m_ij, tgt, 0, dim_size=f_node.size(0), reduce='sum')
        h_j = self.out(torch.cat([f_node, m_j], dim=-1))
        h_j = self.dropout(h_j)
    
        return h_j, m_ij