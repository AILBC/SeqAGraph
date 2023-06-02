import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from torch.utils.tensorboard.writer import SummaryWriter
from .batch_loader import ReactionData
from .transformer_gnn_v2 import GraphEncoder
from .at_decoder import SmilesDecoder
from .inference.huggingface_infer import Beam_Generate as Huggingface_Beam
from .model_utils import embedding_init, linear_init, node_padding, get_mask, AbsPositionalEmbedding, CrossEntropy

def get_deepnorm(enc_layer: Optional[int]=0, dec_layer: Optional[int]=0):
    """
    calculate deepnorm alpha and beta.
    """
    if enc_layer == 0:
        if dec_layer == 0: return 1., 1.
        else:
            return (3 * dec_layer) ** (1 / 4), (12 * dec_layer)** (-1 / 4)
    else:
        return 0.81 * (((enc_layer ** 4) * dec_layer) ** (1 / 16)), 0.87 * (((enc_layer ** 4) * dec_layer) ** (-1 / 16))


class SMILEdit(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        K: int,
        enc_layer: int,
        dec_layer: int,
        enc_head: int,
        dec_head: int,
        dropout: float,
        max_bond_count: int,
        max_dist_count: int,
        max_dist: int,
        max_deg: int,
        vocab: dict[str:int],
        task: Optional[str]='dualtask',
        reaction_class: Optional[bool]=False,
        pe_type: Optional[str]=None,
        ffn_type: Optional[str]='glu',
        norm_type: Optional[str]='rmsnorm',
        labelsmooth: Optional[float]=0.1,
        gamma: Optional[float]=0.0,
        augment_N: Optional[int]=2,
        max_perm_idx: Optional[int]=10,
        device: Optional[str]='cuda:0'
    ):
        super(SMILEdit, self).__init__()
        self.vocab = vocab
        self.rvocab = {id:t for t, id in self.vocab.items()}
        self.task = task
        self.d_model = d_model
        self.labelsmooth = labelsmooth
        self.ignore_len = 2 # class, dualtask
        self.augment_N = augment_N
        self.forward_count = 0

        self.vocab_emb = nn.Embedding(len(self.vocab), d_model, padding_idx=self.vocab['<PAD>'])
        self.class_emb = nn.Embedding(11, d_model, padding_idx=0)
        self.task_emb = nn.Embedding(2, d_model)
        self.predict = nn.ModuleDict()
        if self.task in ['retrosynthesis', 'dualtask']:
            self.predict['retro'] = nn.Linear(d_model, len(self.vocab))
        if self.task in ['forwardsynthesis', 'dualtask']:
            self.predict['fwd'] = nn.Linear(d_model, len(self.vocab))
        self.dropout = nn.Dropout(dropout)
        self.positionalemb = AbsPositionalEmbedding(
            d_model=d_model,
            max_len=2048,
            dropout=dropout
        )
        enc_alpha, enc_beta = get_deepnorm(enc_layer, dec_layer)
        self.graphencoder = GraphEncoder(
            d_model=d_model,
            d_ff=d_ff,
            layer=enc_layer,
            head=enc_head,
            dropout=dropout,
            K=K,
            max_bond_count=max_bond_count,
            max_dist_count=max_dist_count,
            max_dist=max_dist,
            max_deg=max_deg,
            reaction_class=reaction_class,
            ffn_type=ffn_type,
            norm_type=norm_type,
            task=task,
            pe_type=pe_type,
            sin_emb=self.positionalemb.pe,
            init_gain=enc_beta,
            residual_scale=enc_alpha,
            ignore_len=self.ignore_len
        )
        dec_alpha, dec_beta = get_deepnorm(0, dec_layer)
        self.decoder = SmilesDecoder(
            d_model=d_model,
            d_ff=d_ff,
            layer=dec_layer,
            head=dec_head,
            dropout=dropout,
            task=task,
            pe_type=pe_type,
            sin_emb=self.positionalemb.pe,
            ffn_type=ffn_type,
            norm_type=norm_type,
            init_gain=dec_beta,
            residual_scale=dec_alpha,
            ignore_len=self.ignore_len
        )
        self.criterion = CrossEntropy(
            vocab=self.vocab,
            gamma=gamma,
            ignore_idx=self.vocab['<PAD>']
        )

        self.init_param()
    
    def init_param(self):
        self.vocab_emb = embedding_init(self.vocab_emb)
        self.class_emb = embedding_init(self.class_emb)
        self.task_emb = embedding_init(self.task_emb)
        for key in self.predict.keys():
            self.predict[key] = linear_init(self.predict[key])
    
    def task_project(self, x: torch.Tensor, num_list: list[int]):
        if self.task == 'dualtask':
            x_in = torch.split(x, num_list, dim=0)
            x_out = torch.cat([self.predict['retro'](x_in[0]), self.predict['fwd'](x_in[1])], dim=0)
        elif self.task == 'retrosynthesis':
            x_out = self.predict['retro'](x)
        elif self.task == 'forwardsynthesis':
            x_out = self.predict['fwd'](x)
        return x_out
    
    def find_length(self, x: torch.Tensor, max_len: Optional[int]=0):
        x_idx = torch.argmax(x, dim=-1)
        if max_len <= 0: max_len = x_idx.size(-1)
        x_len = torch.full((x.size(0),), max_len, dtype=torch.long, device=x.device)
        row, col = torch.where(x_idx == self.vocab['<EOS>'])
        row, row_idx = row.unique(return_inverse=True, dim=-1)
        row_idx = row_idx.unique(dim=-1)
        col = col.index_select(0, row_idx)
        x_len[row] = col + 1
        x_len[x_len > max_len] = max_len
        return x_len
    
    def get_accuracy(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        f_task: torch.Tensor
    ):
        retro_acc = {'token': 0., 'seq': 0.}
        forward_acc = {'token': 0., 'seq': 0.}
        with torch.no_grad():
            predict = x.softmax(dim=-1).argmax(dim=-1)
            padding_idx = target.eq(self.vocab['<PAD>'])
            token_acc = ((predict == target).to(torch.float) * (~padding_idx)).sum(dim=-1)\
                / (~padding_idx).sum(dim=-1)
            seq_acc = torch.logical_or((predict == target), padding_idx)\
                .all(dim=-1).to(torch.float)
            if f_task.eq(0).any():
                retro_acc['token'] = token_acc[f_task.eq(0)].mean().item()
                retro_acc['seq'] = seq_acc[f_task.eq(0)].mean().item()
            if f_task.eq(1).any():
                forward_acc['token'] = token_acc[f_task.eq(1)].mean().item()
                forward_acc['seq'] = seq_acc[f_task.eq(1)].mean().item()
        return retro_acc, forward_acc
    
    def get_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        num_list: list[int],
        smooth: Optional[float]=0.1
    ):
        retro_x, fwd_x = x.split(num_list, 0)
        retro_tgt, fwd_tgt = target.split(num_list, 0)
        retro_loss, fwd_loss, loss_count = 0., 0., 0
        if len(retro_x) > 0:
            retro_loss = self.criterion(retro_x.transpose(1, 2), retro_tgt, smooth).sum(dim=-1).mean()
            loss_count += 1
        if len(fwd_x) > 0:
            fwd_loss = self.criterion(fwd_x.transpose(1, 2), fwd_tgt, smooth).sum(dim=-1).mean()
            loss_count += 1
        loss = (retro_loss + fwd_loss) / loss_count
        return loss
    
    def forward(self, data: ReactionData, writer: Optional[SummaryWriter]=None):
        batch_size = len(data.tgt_len)
        device = data.tgt_len.device
        num_list = torch.bincount(data.f_task, minlength=2).tolist()
        class_emb, task_emb = self.class_emb(data.f_class).unsqueeze(1),\
            self.task_emb(data.f_task).unsqueeze(1)
        begin = torch.full((batch_size, 1), self.vocab['<BOS>'],
                dtype=torch.long, device=data.tgt_len.device) #start
        begin = self.vocab_emb(begin)

        graph_emb = self.graphencoder(
            data=data
        )

        graph_emb = node_padding(
            node=graph_emb,
            node_num=data.graph_len
        )
        graph_len = data.graph_len

        self.decoder.init_cache()
        self.decoder.update_cache(
            cache_type=['graph'],
            cache=graph_emb
        )
        
        tgt_emb = self.vocab_emb(data.tgt)
        tgt_emb = torch.cat([class_emb, task_emb, begin, tgt_emb], dim=1) * math.sqrt(self.d_model)
        tgt_len = data.tgt_len + self.ignore_len + 1
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = get_mask(
            length=tgt_len,
            q_len=tgt_emb.size(1),
            k_len=tgt_emb.size(1),
            batch_size=batch_size,
            ignore_len=self.ignore_len,
            causal=True,
            device=device
        )
        context_mask = get_mask(
            length=graph_len,
            q_len=tgt_emb.size(1),
            k_len=graph_emb.size(1),
            batch_size=batch_size,
            ignore_len=self.ignore_len,
            device=device
        )
        tgt_emb = self.decoder(
            x=tgt_emb,
            causal_mask=causal_mask,
            context_mask=context_mask,
            num_list=num_list
        )
        tgt_emb = tgt_emb[:, self.ignore_len:-1]
        tgt_out = self.task_project(tgt_emb, num_list)
        
        loss = self.get_loss(tgt_out, data.tgt, num_list, self.labelsmooth)

        retro_acc, forward_acc = self.get_accuracy(
            x=tgt_out,
            target=data.tgt,
            f_task=data.f_task
        )
        self.forward_count += 1
        return loss, retro_acc, forward_acc

    def search(
        self,
        data: ReactionData,
        beam_size: int,
        max_step: int,
        T: Optional[float]=1.0,
        beam_group: Optional[int]=1,
        top_k: Optional[int]=0,
        top_p: Optional[float]=0.,
        cano_search: Optional[bool]=False
    ):
        batch_size = len(data.graph_len)
        device = data.graph_len.device
        num_list = torch.bincount(data.f_task, minlength=2).tolist()
        class_emb, task_emb = self.class_emb(data.f_class).unsqueeze(1),\
            self.task_emb(data.f_task).unsqueeze(1)
        begin = torch.full((batch_size, 1), self.vocab['<BOS>'],
                dtype=torch.long, device=device) #start
        begin = self.vocab_emb(begin)

        graph_emb = self.graphencoder(
            data=data
        )

        graph_emb = node_padding(
            node=graph_emb,
            node_num=data.graph_len
        )
        self.decoder.init_cache()
        
        beam_search = Huggingface_Beam(
            beam_size=beam_size,
            batch_size=batch_size,
            bos_token_ids=self.vocab['<BOS>'],
            pad_token_ids=self.vocab['<PAD>'],
            eos_token_ids=self.vocab['<EOS>'],
            vocab=self.vocab,
            rvocab=self.rvocab,
            length_penalty=0.,
            min_len=1,
            max_len=max_step,
            beam_group=beam_group,
            temperature=T,
            top_k=top_k,
            top_p=top_p,
            return_num=10,
            remove_finish_batch=True,
            device=device
        )

        graph_emb = graph_emb.repeat_interleave(beam_size, 0)
        class_emb = class_emb.repeat_interleave(beam_size, 0)
        task_emb = task_emb.repeat_interleave(beam_size, 0)
        graph_len = data.graph_len.repeat_interleave(beam_size, 0)
        f_task = data.f_task.repeat_interleave(beam_size, 0)
        num_list = torch.bincount(f_task, minlength=2).tolist()
        self.decoder.update_cache(
            cache_type=['graph'],
            cache=graph_emb
        )

        context_mask = get_mask(
            length=graph_len,
            q_len=self.ignore_len + 1,
            k_len=graph_emb.size(1),
            batch_size=graph_len.size(0),
            ignore_len=self.ignore_len,
            device=device
        )
        
        for i in range(max_step):
            x_in = beam_search.current_token.reshape(-1, 1).to(device)
            x_in = self.vocab_emb(x_in)
            if i == 0:
                x_in = torch.cat([class_emb, task_emb, x_in], dim=1)
            x_in = x_in * math.sqrt(self.d_model)

            x_out = self.decoder(
                x=x_in,
                causal_mask=None,
                context_mask=context_mask,
                num_list=num_list,
                step=i
            )
            if i == 0: x_out = x_out[:, -1].unsqueeze(1)
            x_out = self.task_project(x_out, num_list)
            beam_search.generate(x_out)
            if beam_search.is_done: break
            unfinish_idx = beam_search.mem_ids.to(device)
            self.decoder.update_cache(
                cache_type=['last', 'graph'],
                idx=unfinish_idx
            )
            if i == 0:
                context_mask = context_mask[:, :, :1]
            f_task = f_task.index_select(0, unfinish_idx)
            context_mask = context_mask.index_select(0, unfinish_idx)
            num_list = torch.bincount(f_task, minlength=2).tolist()
        if cano_search:
            return beam_search.finish_generate_with_cano()
        else:
            return beam_search.finish_generate()