import numpy as np
import torch
import re

from .infer_base import BeamGenerateBase
from .huggingface_beamsearch import BeamSearchScorer
from model.preprocess.data_utils import canonicalize_smiles

PATTERN = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
REGEX = re.compile(PATTERN)

class Beam_Generate(BeamGenerateBase):
    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        bos_token_ids: int,
        pad_token_ids: int,
        eos_token_ids: int,
        vocab: dict[str:int],
        rvocab: dict[int:str],
        length_penalty=1.,
        min_len=1,
        max_len=256,
        beam_group=1,
        temperature=1.,
        top_k=0,
        top_p=0.,
        return_num=10,
        remove_finish_batch=True,
        device='cuda:0'
    ):
        super(Beam_Generate, self).__init__(
            beam_size=beam_size,
            batch_size=batch_size,
            bos_token_ids=bos_token_ids,
            pad_token_ids=pad_token_ids,
            eos_token_ids=eos_token_ids,
            length_penalty=length_penalty,
            min_len=min_len,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_num=return_num,
            device=device
        )
        self.beam_group = beam_group
        self.remove_finish_batch = remove_finish_batch
        self.vocab = vocab
        self.rvocab = rvocab

        assert self.beam_size % self.beam_group == 0
        self.beam_per_group = self.beam_size // self.beam_group
        self.beam_search = BeamSearchScorer(
            batch_size=self.batch_size,
            num_beams=self.beam_size,
            device=self.device,
            length_penalty=self.length_penalty,
            do_early_stopping=False,
            num_beam_hyps_to_keep=self.beam_size,
            num_beam_groups=self.beam_group
        )
        self.last_alive = self.unfinish_batch.tolist()
        self.ids_table = torch.arange(self.batch_size * self.beam_size, dtype=torch.long,
                                      device=self.device).reshape(self.batch_size, self.beam_size)

        self._prepare()

    @property
    def is_done(self) -> bool:
        return (self.beam_search.is_done) or (self.all_token.size(-1) >= self.max_len)

    @property
    def unfinish_batch(self) -> torch.Tensor:
        return torch.logical_not(self.beam_search._done)

    @property
    def current_token(self) -> torch.Tensor:
        if self.remove_finish_batch:
            cur_token = self.cur_token.reshape(self.batch_size, self.beam_size)[
                self.unfinish_batch].reshape(-1)
        else:
            cur_token = self.cur_token.reshape(-1)
        return cur_token

    @property
    def mem_ids(self) -> torch.Tensor:
        if self.remove_finish_batch:
            mem_ids = self.beam_idx.reshape(self.batch_size, self.beam_size)[
                self.unfinish_batch]
            mem_ids = mem_ids.reshape(-1)
            if sum(self.last_alive) < self.batch_size:
                self.ids_table[self.last_alive] = torch.arange(sum(self.last_alive) * self.beam_size,
                                                               dtype=torch.long, device=self.device).reshape(-1, self.beam_size)
                mem_ids = self.ids_table.reshape(-1).index_select(dim=0, index=mem_ids)
            self.last_alive = self.unfinish_batch.tolist()
        else:
            mem_ids = self.beam_idx.reshape(-1)
        return mem_ids

    def _prepare(
        self
    ):
        self.scores = np.array([], dtype=np.float32)
        self.beam_scores = torch.full((self.batch_size, self.beam_size), -float('inf'), dtype=torch.float, device=self.device)
        self.beam_scores[:, ::self.beam_per_group] = 0.
        self.beam_scores = self.beam_scores.view((self.batch_size * self.beam_size,))
        self.cur_token = torch.full((self.batch_size * self.beam_size,), self.bos_token_ids, dtype=torch.long, device=self.device)
        self.all_token = self.cur_token.clone().reshape(-1, 1)

        self.batchid_list = [i // self.beam_size for i in range(self.batch_size * self.beam_size)]
        self.groupid_list = []
        for i in range(self.batch_size):
            self.groupid_list.extend([j // self.beam_per_group for j in range(self.beam_size)])
        self.beam_idx = torch.arange(self.batch_size * self.beam_size, dtype=torch.long, device=self.device)

    def _scores_process(
        self,
        # each token's logits score, size(batch * beam, vocab_size)
        scores: torch.Tensor
    ):
        cur_len = self.all_token.size(-1)
        if cur_len < self.min_len:
            scores[:, self.eos_token_ids] = -float('inf')
        return scores

    def _sample_process(
        self,
        # each sequence's sum logits score, size(batch * beam, vocab_size)
        scores: torch.Tensor,
        min_keep=2
    ):
        if self.top_k > 0:
            top_k = min(max(self.top_k, min_keep), scores.size(-1))
            # select the lowest score in topk
            remove_ids = scores < torch.topk(scores, top_k, dim=-1)[0][..., -1, None]
            scores = scores.masked_fill(remove_ids, -float('inf'))
        if self.top_p > 0.:
            sorted_logits, sorted_ids = torch.sort(scores, descending=True)
            cumsum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # cumsum calculate the sum of each scores, which can easily find the threshold
            remove_sorted_ids = cumsum_probs > self.top_p
            # the next three step to finish the min_keep, then accept the threshold and the first token.
            # if min_keep > 1:
            remove_sorted_ids[..., :min_keep] = False
            # remove_sorted_ids[..., 1:] = remove_sorted_ids[..., :-1].clone()
            # remove_sorted_ids[..., 0] = False
            remove_ids = remove_sorted_ids.scatter(dim=1, index=sorted_ids, src=remove_sorted_ids)
            scores = scores.masked_fill(remove_ids, -float('inf'))
        return scores

    def _finished_batch_pad(
        self,
        # the decoder output like [unfinish_batch * beam, vocab_size], before log_softmax
        dec_output: torch.Tensor
    ):
        if dec_output.size(0) < self.batch_size * self.beam_size:
            new_dec_output = torch.zeros((self.batch_size, self.beam_size * dec_output.size(-1)),
                dtype=dec_output.dtype, device=dec_output.device)
            new_dec_output[self.unfinish_batch] = dec_output.reshape(
                -1, self.beam_size * dec_output.size(-1))
            return new_dec_output.reshape(self.batch_size * self.beam_size, -1)
        else:
            return dec_output

    def generate(
        self,
        # the decoder output like [unfinish_batch * beam, 1, vocab_size], before log_softmax
        dec_output: torch.FloatTensor,
        lat_prob=None
    ):
        dec_output = dec_output[:, -1, :]
        if self.remove_finish_batch:
            dec_output = self._finished_batch_pad(dec_output)
        # size(batch * beam, vocab_size)
        next_token_logits = (dec_output / self.temperature).log_softmax(dim=-1)
        if lat_prob is not None:
            next_token_logits += lat_prob
        next_token_logits = self._scores_process(next_token_logits)
        next_token_logits = self._sample_process(next_token_logits, min_keep=self.beam_size // self.beam_group) if self.all_token.size(-1) == 1\
            else self._sample_process(next_token_logits)
        next_token_logits = next_token_logits + self.beam_scores[:, None].expand_as(next_token_logits)

        vocab_size = next_token_logits.size(-1)
        next_token_logits = next_token_logits.view(self.batch_size, self.beam_group, self.beam_per_group * vocab_size)
        next_token_probs = next_token_logits.softmax(dim=-1)

        for group_id in range(self.beam_group):
            group_idx = [group_id == i for i in self.groupid_list]
            group_token = self.all_token[group_idx]
            group_logits = next_token_logits[:, group_id]
            group_probs = next_token_probs[:, group_id]
            if self.top_k > 0 or self.top_p > 0.:
                next_tokens = torch.multinomial(group_probs, num_samples=self.beam_per_group * 2)
                next_token_scores = torch.gather(group_logits, dim=-1, index=next_tokens)
                next_token_scores, scores_ids = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, dim=-1, index=scores_ids)
            else:
                next_token_scores, next_tokens = torch.topk(
                    group_logits, self.beam_per_group * 2, dim=1, largest=True, sorted=True)

            next_ids, next_tokens = next_tokens.div(vocab_size, rounding_mode='floor'), next_tokens.fmod(vocab_size)
            beam_output = self.beam_search.process(
                input_ids=group_token,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_ids,
                pad_token_id=self.pad_token_ids,
                eos_token_id=self.eos_token_ids
            )
            self.beam_scores[group_idx] = beam_output['next_beam_scores']
            self.all_token[group_idx] = group_token[beam_output['next_beam_indices']]
            group_token = torch.cat([group_token[beam_output['next_beam_indices']],
                                    beam_output['next_beam_tokens'].unsqueeze(-1)], dim=-1)
            self.cur_token[group_idx] = group_token[:, -1]
            self.beam_idx[group_idx] = torch.div(beam_output['next_beam_indices'], self.beam_per_group, rounding_mode='floor') * self.beam_size + \
                group_id * self.beam_per_group + \
                torch.fmod(beam_output['next_beam_indices'], self.beam_per_group)

        self.all_token = torch.cat([self.all_token, self.cur_token.unsqueeze(-1)], dim=-1)

    def finish_generate(
        self
    ):
        beam_output = self.beam_search.finalize(
            input_ids=self.all_token,
            final_beam_scores=self.beam_scores,
            final_beam_tokens=None,
            final_beam_indices=None,
            max_length=self.max_len,
            pad_token_id=self.pad_token_ids,
            eos_token_id=self.eos_token_ids
        )
        seq_result = beam_output['sequences'].reshape(self.batch_size, self.beam_size, -1)[..., 1:]  # ignore BOS token
        seq_scores = beam_output['sequence_scores'].reshape(self.batch_size, self.beam_size)
        seq_result, seq_scores = seq_result.detach().cpu(), seq_scores.detach().cpu()
        
        if self.beam_group > 1:
            new_seq_result = -1 * torch.ones((self.batch_size, self.return_num, seq_result.size(-1)), dtype=torch.long)
            new_seq_scores = torch.zeros((self.batch_size, self.return_num), dtype=torch.float)
            
            for batch_id in range(self.batch_size):
                batch_result, batch_scores = seq_result[batch_id], seq_scores[batch_id]
                beam_count = 0
                beam_result_cache = torch.zeros((self.beam_size, seq_result.size(-1)), dtype=torch.long)
                beam_freq_cache, beam_scores_cache = torch.zeros((self.beam_size,), dtype=torch.long),\
                    torch.zeros((self.beam_size,), dtype=torch.float)

                for beam_id in range(self.beam_size):
                    beam_result, beam_scores = batch_result[beam_id], batch_scores[beam_id]
                    # ignore the exist repeat beam result
                    # if beam_result_cache.eq(beam_result).min(dim=-1)[0].any():
                    #     continue
                    if beam_result_cache.eq(beam_result).all(dim=-1).any():
                        continue
                    # beam_index = batch_result.eq(beam_result).min(dim=-1)[0]
                    beam_index = batch_result.eq(beam_result).all(dim=-1)
                    # beam_freq_cache[beam_count] = beam_index.sum()
                    beam_freq_cache[beam_count] = 1
                    beam_scores_cache[beam_count] = batch_scores[beam_index].max()
                    beam_result_cache[beam_count] = beam_result
                    beam_count += 1
                
                # sort according to frequency and mean scores
                beam_freq_cache = beam_freq_cache + (1 / (-beam_scores_cache + 1))
                rerank_scores, rerank_idx = torch.topk(beam_freq_cache, k=self.return_num, dim=-1)
                new_seq_result[batch_id] = beam_result_cache[rerank_idx]
                new_seq_scores[batch_id] = rerank_scores
        
        else:
            new_seq_result = seq_result[:, :self.return_num]
            new_seq_scores = seq_scores[:, :self.return_num]

        return new_seq_result, new_seq_scores
    
    def finish_generate_with_cano(
        self
    ):
        beam_output = self.beam_search.finalize(
            input_ids=self.all_token,
            final_beam_scores=self.beam_scores,
            final_beam_tokens=None,
            final_beam_indices=None,
            max_length=self.max_len,
            pad_token_id=self.pad_token_ids,
            eos_token_id=self.eos_token_ids
        )
        seq_result = beam_output['sequences'].reshape(self.batch_size, self.beam_size, -1)[..., 1:]  # ignore BOS token
        seq_scores = beam_output['sequence_scores'].reshape(self.batch_size, self.beam_size)
        seq_result, seq_scores = seq_result.detach().cpu(), seq_scores.detach().cpu()
        seq_result_numpy = seq_result.numpy()
        seq_len = seq_result.size(-1)
        seq_result_cano = torch.zeros_like(seq_result)
        for batch_id in range(self.batch_size):
            for beam_id in range(self.beam_size):
                uncano_result = seq_result_numpy[batch_id, beam_id]
                if (uncano_result == self.eos_token_ids).sum() > 0:
                    eos_idx = np.where(uncano_result == self.eos_token_ids)[0][0]
                    uncano_result = uncano_result[:eos_idx]
                uncano_result = [self.rvocab[_] for _ in uncano_result]
                uncano_result = ''.join(uncano_result)
                cano_result, valid = canonicalize_smiles(uncano_result, True, False, False)
                if not valid: seq_scores[batch_id, beam_id] += (-1e5)
                cano_result = [token for token in REGEX.findall(cano_result)]
                cano_result = [self.vocab.get(token, self.pad_token_ids) for token in cano_result][:seq_len - 1]
                cano_result = torch.tensor(cano_result + [self.eos_token_ids] + [self.pad_token_ids] * (seq_len - len(cano_result) - 1), dtype=torch.long)
                seq_result_cano[batch_id, beam_id] = cano_result
        
        new_seq_result = -1 * torch.ones((self.batch_size, self.return_num, seq_result_cano.size(-1)), dtype=torch.long)
        new_seq_scores = torch.zeros((self.batch_size, self.return_num), dtype=torch.float)
        
        for batch_id in range(self.batch_size):
            batch_result, batch_scores = seq_result_cano[batch_id], seq_scores[batch_id]
            beam_count = 0
            beam_result_cache = torch.zeros((self.beam_size, seq_result_cano.size(-1)), dtype=torch.long)
            beam_freq_cache, beam_scores_cache = torch.zeros((self.beam_size,), dtype=torch.long),\
                torch.zeros((self.beam_size,), dtype=torch.float)

            for beam_id in range(self.beam_size):
                beam_result, beam_scores = batch_result[beam_id], batch_scores[beam_id]
                # ignore the exist repeat beam result
                # if beam_result_cache.eq(beam_result).min(dim=-1)[0].any():
                #     continue
                if beam_result_cache.eq(beam_result).all(dim=-1).any():
                    continue
                # beam_index = batch_result.eq(beam_result).min(dim=-1)[0]
                beam_index = batch_result.eq(beam_result).all(dim=-1)
                beam_freq_cache[beam_count] = beam_index.sum()
                # beam_freq_cache[beam_count] = 1
                beam_scores_cache[beam_count] = batch_scores[beam_index].mean()
                beam_result_cache[beam_count] = beam_result
                beam_count += 1
            
            # sort according to frequency and mean scores
            beam_freq_cache = beam_freq_cache + (1 / (-beam_scores_cache + 1))
            rerank_scores, rerank_idx = torch.topk(beam_freq_cache, k=self.return_num, dim=-1)
            new_seq_result[batch_id] = beam_result_cache[rerank_idx]
            new_seq_scores[batch_id] = rerank_scores

        return new_seq_result, new_seq_scores
