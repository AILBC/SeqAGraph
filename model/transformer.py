import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from .model_utils import linear_init, Attention

class SequenceAttention(Attention):
    def __init__(
        self,
        d_model: int,
        head: int,
        dropout: float,
        attention_type: Optional[str]='self',
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
        init_gain: Optional[float]=1.,
        ignore_len: Optional[int]=0
    ):
        assert d_model % head == 0
        super(SequenceAttention, self).__init__(
            d_head=math.ceil(d_model / head),
            pe_type=pe_type,
            sin_emb=sin_emb
        )
        self.d_model = d_model
        self.head = head
        self.attention_type = attention_type
        self.ignore_len = ignore_len

        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

        self.dropout = dropout

        self.init_param(gain=init_gain)
    
    def init_param(self, gain=1.):
        self.Wq = linear_init(self.Wq, zero_bias=True)
        self.Wk = linear_init(self.Wk, zero_bias=True)
        self.Wv = linear_init(self.Wv, zero_bias=True, gain=gain)
        self.out = linear_init(self.out, zero_bias=True, gain=gain)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor]=None,
        value: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None, # size(bsz, 1, q_len, k_len), True means the value will take part in attention
        step: Optional[int]=0
    ):
        batch_size = query.size(0)
        if key is None: key = query
        if value is None: value = query
        
        def shape(x: torch.Tensor):
            return x.reshape(batch_size, -1, self.head, self.d_head).transpose(1, 2)

        def unshape(x: torch.Tensor):
            return x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.head * self.d_head)
        
        q = self.Wq(query)
        k = self.Wk(key)
        v = self.Wv(value)
        if self.pe_type == 'rope':
            if self.attention_type == 'self':
                if self.ignore_len > 0 and step == 0:
                    q_emb, k_emb = self.compute_rope([q[:, self.ignore_len:], k[:, self.ignore_len:]], [step, 0])
                    q, k = torch.cat([q[:, :self.ignore_len], q_emb], dim=1), torch.cat([k[:, :self.ignore_len], k_emb], dim=1)
                elif self.ignore_len > 0 and step > 0:
                    q, k_emb = self.compute_rope([q, k[:, self.ignore_len:]], [step, 0])
                    k = torch.cat([k[:, :self.ignore_len], k_emb], dim=1)
                else:
                    q, k = self.compute_rope([q, k], [step, 0])
            elif self.attention_type == 'context':
                if self.ignore_len > 0 and step == 0:
                    q_emb, k = self.compute_rope([q[:, self.ignore_len:], k], [step, 0])
                    q = torch.cat([q[:, :self.ignore_len], q_emb], dim=1)
                else:
                    q, k = self.compute_rope([q, k], [step, 0])

        q, k, v = shape(q), shape(k), shape(v)
        
        out = F.scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = unshape(out)        
        out = self.out(out)
        return out
    

class MultiAttention(Attention):
    def __init__(
        self,
        d_model: int,
        head: int,
        dropout: float,
        attention_type: Optional[str]='self',
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
        init_gain: Optional[float]=1.,
        ignore_len: Optional[int]=0
    ):
        assert d_model % head == 0
        self.head = head
        super(MultiAttention, self).__init__(
            d_head=math.ceil(d_model / self.head),
            pe_type=pe_type,
            sin_emb=sin_emb
        )
        self.d_model = d_model
        self.attention_type = attention_type
        self.ignore_len = ignore_len

        self.Wq = nn.Linear(self.d_model, self.d_model * 2)
        self.Wk = nn.Linear(self.d_model, self.d_model * 2)
        self.Wv = nn.Linear(self.d_model, self.d_model * 2)
        self.out = nn.Linear(self.d_model * 2, self.d_model)

        self.dropout = dropout

        self.init_param(gain=init_gain)
    
    def init_param(self, gain=1.):
        self.Wq = linear_init(self.Wq, zero_bias=True)
        self.Wk = linear_init(self.Wk, zero_bias=True)
        self.Wv = linear_init(self.Wv, zero_bias=True, gain=gain)
        self.out = linear_init(self.out, zero_bias=True, gain=gain)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor]=None,
        value: Optional[torch.Tensor]=None,
        mask1: Optional[torch.Tensor]=None, # size(bsz, 1, q_len, k_len), True means the value will take part in attention
        mask2: Optional[torch.Tensor]=None,
        step: Optional[int]=0
    ):
        batch_size = query.size(0)
        if key is None: key = query
        if value is None: value = query
        
        def shape(x: torch.Tensor):
            return x.reshape(batch_size, -1, self.head, self.d_head).transpose(1, 2)

        def unshape(x: torch.Tensor):
            return x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.head * self.d_head)
        
        q1, q2 = self.Wq(query).chunk(2, -1)
        k1, k2 = self.Wk(key).chunk(2, -1)
        v1, v2 = self.Wv(value).chunk(2, -1)

        if self.pe_type == 'rope':
            if self.ignore_len > 0 and step == 0:
                q1_emb, k1_emb, v1_emb = self.compute_rope([q1[:, self.ignore_len:], k1[:, self.ignore_len:], v1[:, self.ignore_len:]], [step, 0, 0])
                q1, k1, v1 = torch.cat([q1[:, :self.ignore_len], q1_emb], dim=1), torch.cat([k1[:, :self.ignore_len], k1_emb], dim=1), torch.cat([v1[:, :self.ignore_len], v1_emb], dim=1)
                q2_emb = self.compute_rope([q2[:, self.ignore_len:]], [step])[0]
                q2 = torch.cat([q2[:, :self.ignore_len], q2_emb], dim=1)
            elif self.ignore_len > 0 and step > 0:
                q1, k1_emb, v1_emb = self.compute_rope([q1, k1[:, self.ignore_len:], v1[:, self.ignore_len:]], [step, 0, 0])
                k1, v1 = torch.cat([k1[:, :self.ignore_len], k1_emb], dim=1), torch.cat([v1[:, :self.ignore_len], v1_emb], dim=1)
                q2 = self.compute_rope([q2], [step])[0]
            else:
                q1, k1, v1 = self.compute_rope([q1, k1, v1], [step, 0, 0])
                q2 = self.compute_rope([q2], [step])[0]
        
        q1, q2 = shape(q1), shape(q2)
        k1, k2 = shape(k1), shape(k2)
        v1, v2 = shape(v1), shape(v2)
        
        out1 = F.scaled_dot_product_attention(q1, k1, v1, mask1, self.dropout)
        out2 = F.scaled_dot_product_attention(q2, k2, v2, mask2, self.dropout)
        out1, out2 = unshape(out1), unshape(out2)       
        out = self.out(torch.cat([out1, out2], dim=-1))
        return out