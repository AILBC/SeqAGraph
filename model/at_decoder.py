import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from.transformer import SequenceAttention
from .model_utils import FeedForward, Normalization


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        head: int,
        dropout: float,
        task: Optional[str]='dualtask',
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
        ffn_type: Optional[str]='glu',
        norm_type: Optional[str]='rmsnorm',
        init_gain: Optional[float]=1.,
        residual_scale: Optional[float]=1.,
        ignore_len: Optional[int]=0
    ):
        super(DecoderLayer, self).__init__()
        self.task = task
        self.residual_scale = residual_scale

        self.memattention = SequenceAttention(
            d_model=d_model,
            head=head,
            dropout=dropout,
            attention_type='self',
            pe_type=pe_type,
            sin_emb=sin_emb,
            init_gain=init_gain,
            ignore_len=ignore_len
        )
        self.conattention = SequenceAttention(
            d_model=d_model,
            head=head,
            dropout=dropout,
            attention_type='context',
            pe_type=None,
            sin_emb=None,
            init_gain=init_gain,
            ignore_len=ignore_len
        )
        self.feedforward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            init_gain=init_gain,
            ffn_tpye=ffn_type
        )
        self.norm_t1 = nn.ModuleList()
        self.norm_t2 = nn.ModuleList()
        if self.task in ['retrosynthesis', 'dualtask']:
            for _ in range(3):
                self.norm_t1.append(Normalization(d_model, norm_type=norm_type))
        if self.task in ['forwardsynthesis', 'dualtask']:
            for _ in range(3):
                self.norm_t2.append(Normalization(d_model, norm_type=norm_type))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.init_param()
    
    def init_param(self):
        pass

    def build_cache(self):
        self.cache = {'last': None, 'text': None, 'graph': None}

    def task_norm(self, x: torch.Tensor, norm_id: int, num_list: list[int]):
        if self.task == 'dualtask':
            x_in = torch.split(x, num_list, dim=0)
            x_out = torch.cat([self.norm_t1[norm_id](x_in[0]), self.norm_t2[norm_id](x_in[1])], dim=0)
        elif self.task == 'retrosynthesis':
            x_out = self.norm_t1[norm_id](x)
        elif self.task == 'forwardsynthesis':
            x_out = self.norm_t2[norm_id](x)
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        context_mask: torch.Tensor,
        num_list: list[int],
        step: Optional[int]=0
    ):
        self.cache['last'] = x if self.cache['last'] is None else torch.cat([self.cache['last'], x], dim=1)
        x_in = self.memattention(
            query=x,
            key=self.cache['last'],
            value=self.cache['last'],
            mask=causal_mask,
            step=step
        )
        x_out = self.dropout1(x_in) + x * self.residual_scale
        x_out = self.task_norm(x_out, 0, num_list)
        x_in = self.conattention(
            query=x,
            key=self.cache['graph'],
            value=self.cache['graph'],
            mask=context_mask,
            step=step
        )
        x_out = self.dropout2(x_in) + x_out
        x_out = self.task_norm(x_out, 1, num_list)
        x_out = self.dropout3(self.feedforward(x_out)) + x_out * self.residual_scale
        x_out = self.task_norm(x_out, 2, num_list)
        return x_out


class SmilesDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        layer: int,
        head: int,
        dropout: float,
        task: Optional[str]='dualtask',
        pe_type: Optional[str]=None,
        sin_emb: Optional[torch.Tensor]=None,
        ffn_type: Optional[str]='glu',
        norm_type: Optional[str]='rmsnorm',
        init_gain: Optional[float]=1.,
        residual_scale: Optional[float]=1.,
        ignore_len: Optional[int]=0
    ):
        super(SmilesDecoder, self).__init__()
        self.layer = layer

        self.at_transformer = nn.ModuleList()
        for i in range(self.layer):
            self.at_transformer.append(DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                head=head,
                dropout=dropout,
                task=task,
                pe_type=pe_type,
                sin_emb=sin_emb,
                ffn_type=ffn_type,
                norm_type=norm_type,
                init_gain=init_gain,
                residual_scale=residual_scale,
                ignore_len=ignore_len # reaction_class, task
            ))
        
        self.init_param()

    def init_param(self):
        pass
    
    def init_cache(self):
        for i in range(len(self.at_transformer)):
            self.at_transformer[i].build_cache()

    def update_cache(
        self,
        cache_type: list[str],
        cache: Optional[torch.Tensor]=None,
        idx: Optional[torch.Tensor]=None
    ):
        for i in range(len(self.at_transformer)):
            for key in cache_type:
                if cache is not None:
                    self.at_transformer[i].cache[key] = cache
                elif idx is not None:
                    self.at_transformer[i].cache[key] = \
                        self.at_transformer[i].cache[key].index_select(0, idx)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        context_mask: torch.Tensor,
        num_list: list[int],
        step: Optional[int]=0
    ):
        for i in range(self.layer):
            x = self.at_transformer[i](
                x=x,
                causal_mask=causal_mask,
                context_mask=context_mask,
                num_list=num_list,
                step=step
            )
        return x