import math
import torch
import torch.nn as nn

from index_elemtwise import indexelemwise # custom CUDA kernel for index_select + elemtwise operation
from typing import Optional
from torch_scatter import scatter, scatter_softmax
from .batch_loader import ReactionData
from .kp_mpnn import InitialEmbedding
from .model_utils import embedding_init, linear_init, FeedForward, Normalization

def attention_softmax(
    x: torch.Tensor,
    query_idx: torch.Tensor,
    node_num: int,
    eps=1e-5
):
    x_max = scatter(x, query_idx, 0, dim_size=node_num, reduce='max')
    out = (x - x_max.index_select(0, query_idx)).exp()
    out_sum = scatter(out, query_idx, 0, dim_size=node_num, reduce='sum')
    return out / (out_sum.index_select(0, query_idx) + eps)


class GlobalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        K: int,
        layer: int,
        head: int,
        dropout: float,
        max_bond_count: int,
        max_dist: int,
        max_deg: int,
        init_gain: Optional[float]=1.0
    ):
        super(GlobalAttention, self).__init__()
        assert d_model % head == 0
        self.d_model = d_model
        self.head = head
        self.d_head = math.ceil(d_model / head)

        self.Wq = nn.Linear(self.d_model, self.d_model * 2)
        self.Wk = nn.Linear(self.d_model, self.d_model * 2)
        self.Wv = nn.Linear(self.d_model, self.d_model * 2)
        self.out = nn.Linear(self.d_model * 2, self.d_model)

        self.deg_emb = nn.Embedding(max_deg + 1, self.d_model)
        self.dist_emb = nn.Embedding(max_dist + 1, self.d_model, padding_idx=max_dist)
        self.bond_emb = nn.Linear(self.d_model, self.d_model * 2)
        self.bond_update = nn.Linear(self.d_model * 2, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_param(init_gain)
    
    def init_param(self, gain=1.):
        self.Wq = linear_init(self.Wq, zero_bias=True)
        self.Wk = linear_init(self.Wk, zero_bias=True)
        self.Wv = linear_init(self.Wv, zero_bias=True, gain=gain)
        self.out = linear_init(self.out, zero_bias=True, gain=gain)
        self.deg_emb = embedding_init(self.deg_emb)
        self.dist_emb = embedding_init(self.dist_emb)
        self.bond_emb = linear_init(self.bond_emb)
        self.bond_update = linear_init(self.bond_update)

    def forward(
        self,
        f_node: torch.Tensor,
        deg_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
        f_bond: torch.Tensor,
        bond_idx: torch.Tensor, # include self-loop in [f_bond.size(0):]
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        attention_bond_idx: torch.Tensor,
        bond_split: list[int]
    ):
        node_num = f_node.size(0)
        f_deg = self.deg_emb(deg_matrix).reshape(-1, self.head, self.d_head) #size(N, h, h_dim)
        attention_bond, value_bond = self.bond_emb(f_bond).chunk(2, -1)
        attention_bond = attention_bond.reshape(-1, self.head, self.d_head)
        value_bond = value_bond.reshape(-1, self.head, self.d_head)
        f_dist = self.dist_emb(dist_matrix).reshape(-1, self.head, self.d_head)

        def shape(x: torch.Tensor):
            return x.view(node_num, self.head * 2, self.d_head)

        def unshape(x: torch.Tensor):
            return x.reshape(-1, self.head * self.d_head)

        q = shape(self.Wq(f_node))
        k = shape(self.Wk(f_node))
        v = shape(self.Wv(f_node)) #size(N, h * 2, h_dim)

        q_local, q_global = torch.chunk(q, 2, 1)
        k_local, k_global = torch.chunk(k, 2, 1)
        v_local, v_global = torch.chunk(v, 2, 1)

        # score_local = q_local.index_select(0, bond_idx[1]) * k_local.index_select(0, bond_idx[0])
        score_local = indexelemwise(q_local, k_local, bond_idx[1], bond_idx[0], "mul")
        score_local[:attention_bond.size(0)] += attention_bond # [N_bond:N_bond + N] for node self-loop
        score_local = score_local.sum(dim=-1) / math.sqrt(self.d_head)
        score_local = scatter_softmax(score_local, bond_idx[1], 0, dim_size=score_local.size(0))
        score_local = self.dropout(score_local)
        local_out = v_local.index_select(0, bond_idx[0])
        local_out[:attention_bond.size(0)] += value_bond
        local_out = local_out * score_local.unsqueeze(-1)
        local_out = scatter(local_out, bond_idx[1], 0, dim_size=node_num, reduce='sum')
        local_out = unshape(local_out)

        # score_global = q_global.index_select(0, query_idx) * (k_global + f_deg).index_select(0, key_idx)
        score_global = indexelemwise(q_global, k_global + f_deg, query_idx, key_idx, "mul")
        score_global[attention_bond_idx] += attention_bond
        score_global = score_global + f_dist
        score_global = score_global.sum(dim=-1) / math.sqrt(self.d_head)
        score_global = scatter_softmax(score_global, query_idx, 0, dim_size=score_global.size(0)) 
        score_global = self.dropout(score_global)
        # global_out = v_global.index_select(0, key_idx) * score_global.unsqueeze(-1) #size(N*N, h, h_dim)
        global_out = indexelemwise(v_global, score_global.unsqueeze(-1), key_idx, None, "mul")
        global_out = scatter(global_out, query_idx, 0, dim_size=node_num, reduce='sum')
        global_out = unshape(global_out)
        out = self.out(torch.cat([local_out, global_out], dim=-1))

        bond_out = torch.cat([f_bond, out.index_select(0, bond_idx[0, :attention_bond.size(0)])], dim=-1)
        bond_out = list(self.bond_update(bond_out).split(bond_split, 0))
        return out, bond_out


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        K: int,
        layer: int,
        head: int,
        dropout: float,
        max_bond_count: int,
        max_dist: int,
        max_deg: int,
        ffn_type: Optional[str]='glu',
        norm_type: Optional[str]='rmsnorm',
        init_gain: Optional[float]=1.,
        residual_scale: Optional[float]=1.
    ):
        super(GraphTransformerEncoder, self).__init__()
        self.norm0 = None
        self.residual_scale = residual_scale

        self.attention = GlobalAttention(
            d_model=d_model,
            K=K,
            layer=layer,
            head=head,
            dropout=dropout,
            max_bond_count=max_bond_count,
            max_dist=max_dist,
            max_deg=max_deg,
            init_gain=init_gain
        )
        self.feedforward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            init_gain=init_gain,
            ffn_tpye=ffn_type
        )
        self.norm1 = Normalization(d_model, norm_type=norm_type)
        self.norm2 = Normalization(d_model, norm_type=norm_type)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.init_param()
    
    def init_param(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
        deg_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
        f_bond: torch.Tensor,
        bond_idx: torch.Tensor,
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
        attention_bond_idx: torch.Tensor,
        bond_split: list[int]
    ):
        x_in, bond_out = self.attention(
            f_node=x,
            deg_matrix=deg_matrix,
            dist_matrix=dist_matrix,
            f_bond=f_bond,
            bond_idx=bond_idx,
            query_idx=query_idx,
            key_idx=key_idx,
            attention_bond_idx=attention_bond_idx,
            bond_split=bond_split
        )
        x_out = self.dropout1(x_in) + x * self.residual_scale
        x_out = self.norm1(x_out)
        x_out = self.dropout2(self.feedforward(x_out)) + x_out * self.residual_scale
        x_out = self.norm2(x_out)
        return x_out, bond_out


class GraphEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        layer: int,
        head: int,
        dropout: float,
        K: int,
        max_bond_count: int,
        max_dist_count: int,
        max_dist: int,
        max_deg: int,
        reaction_class: bool,
        ffn_type: Optional[str]='glu',
        norm_type: Optional[str]='rmsnorm',
        task: Optional[str]='dualtask',
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
        init_gain: Optional[float]=1.,
        residual_scale: Optional[float]=1.,
        ignore_len: Optional[int]=0
    ):
        super(GraphEncoder, self).__init__()
        self.layer = layer
        self.K = K

        self.dropout = nn.Dropout(dropout)
        self.init_embedding = InitialEmbedding(
            d_model=d_model,
            K=K,
            max_bond_count=max_bond_count,
            max_dist_count=max_dist_count,
            reaction_class=reaction_class
        )
        self.gnn_transformer = nn.ModuleList()
        for i in range(1, layer + 1, 1):
            self.gnn_transformer.append(GraphTransformerEncoder(
                d_model=d_model,
                d_ff=d_ff,
                K=K,
                layer=i,
                head=head,
                dropout=dropout,
                max_bond_count=max_bond_count,
                max_dist=max_dist,
                max_deg=max_deg,
                ffn_type=ffn_type,
                norm_type=norm_type,
                init_gain=init_gain,
                residual_scale=residual_scale
            ))

        self.init_param()
    
    def init_param(self):
        pass
    
    def forward(
        self,
        data: ReactionData
    ):
        data = self.init_embedding(data)
        self_loop_idx = torch.arange(data.f_atom.size(0), dtype=torch.long, device=data.f_atom.device).repeat(2, 1)
        f_node = data.f_atom.clone()
        f_bond = [bond.clone() for bond in data.f_bond]
        for i in range(1, self.layer + 1, 1):
            f_node, bond_out = self.gnn_transformer[i - 1](
                x=f_node,
                deg_matrix=data.deg,
                dist_matrix=data.dist,
                f_bond=torch.cat(f_bond[:i], dim=0),
                bond_idx=torch.cat([torch.cat(data.bond_idx[:i], dim=-1), self_loop_idx], dim=-1),
                query_idx=data.query_idx,
                key_idx=data.key_idx,
                attention_bond_idx=torch.cat(data.batch_bond_idx[:i], dim=0),
                bond_split=[_.size(0) for _ in data.f_bond[:i]]
            )
            for j in range(len(bond_out)):
                f_bond[j] = bond_out[j].clone()
        return f_node