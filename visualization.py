import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Optional

from parsers import get_parser
from model.preprocess.dataset_base import DATA_DIR
from model.preprocess.data_utils import smi2token, canonicalize_smiles
from model.preprocess.featurize import get_feat
from model.batch_loader import ReactionData, ReactionDataset
from model.prediction_model import SMILEdit
from model.model_utils import CKPT_PATH, setseed, ModelSave


class OneStepPrediction():
    def __init__(self, args):
        assert args.mode == 'eval'

        vocab_path = os.path.join(DATA_DIR, f'vocabulary({args.vocab_name}).txt')
        token, count = [], []
        with open(vocab_path, 'r') as f:
            for data in f:
                t, c = data.strip().split('\t')
                token.append(t)
                count.append(int(c))
        extra_vocab = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        token.extend(extra_vocab)
        self.vocab = {t:id for id, t in enumerate(token)}
        self.rvocab = {id:t for id, t in enumerate(token)}

        self.model = SMILEdit(
            d_model=args.d_model,
            d_ff=args.d_ff,
            K=args.K,
            enc_layer=args.enc_layer,
            dec_layer=args.dec_layer,
            enc_head=args.enc_head,
            dec_head=args.dec_head,
            dropout=args.dropout,
            max_bond_count=args.max_bond_count,
            max_dist_count=args.max_dist_count,
            max_dist=len(args.dist_block),
            max_deg=args.max_deg,
            vocab=self.vocab,
            task=args.task,
            reaction_class=args.reaction_class,
            pe_type=args.pe_type,
            ffn_type=args.ffn_type,
            norm_type=args.norm_type,
            labelsmooth=args.labelsmooth,
            gamma=args.gamma,
            alpha=args.alpha,
            augment_N=args.augment_N,
            max_perm_idx=args.max_perm_idx,
            noatom_idx=[],
            device=args.device
        )
        ckpt_dir = os.path.join(CKPT_PATH, args.save_name)
        ckpt_dir = os.path.join(ckpt_dir, args.ckpt_path)
        model_save = ModelSave(
            ckpt_dir=ckpt_dir,
            const_save=[],
            w1=0.9
        )
        self.model = model_save.load(
            model_name=args.ckpt_name[0],
            model=self.model,
            device=args.device
        )
        self.model = self.model.to(args.device)
        self.args = args
        if args.eval_task == 'retrosynthesis':
            self.eval_task = 0
        elif args.eval_task == 'forwardsynthesis':
            self.eval_task = 1
        else:
            raise ValueError('eval task must in retrosynthesis or forwardsynthesis')

    def preprocess(self, smi_dir: str):
        raw_smi, graph, graph_len = [], [], []

        with open(smi_dir, 'r') as f:
            for data in f:
                data = data.strip('\n')
                smi, valid = canonicalize_smiles(
                    smi=data,
                    retain_smi=True
                )
                smi_graph = get_feat(
                    token=smi,
                    vocab=self.vocab,
                    dual_task=self.eval_task,
                    max_deg=self.args.max_deg,
                    K=self.args.K,
                    kernel=self.args.kernel,
                    max_bond_count=self.args.max_bond_count,
                    max_dist_count=self.args.max_dist_count
                )
                raw_smi.append(smi)
                graph.append(smi_graph)
                graph_len.append(smi_graph['atom_feat'].shape[0])
        
        print(f"load {len(raw_smi)} molecules, task: {self.args.eval_task}, model: {self.args.ckpt_name[0]}.pt")

        batch_size = len(graph_len)
        batch_data = ReactionData()
        batch_data.graph_len = torch.tensor(graph_len, dtype=torch.long)
        batch_data.seq = torch.tensor([])
        batch_data.seq_len = torch.tensor([])
        batch_data.seq_aug = torch.tensor([])
        batch_data.seq_suf = torch.tensor([])
        batch_data.tgt = torch.tensor([])
        batch_data.tgt_len = torch.tensor([])
        batch_data.tgt_aug = torch.tensor([])
        batch_data.tgt_suf = torch.tensor([])
        batch_data.f_class = torch.full((batch_size,), 0, dtype=torch.long)
        batch_data.f_task = torch.tensor([self.eval_task] * batch_size, dtype=torch.long)

        batch_dist = []
        batch_deg = []
        batch_atom_feat = []
        batch_bond_idx = [[] for _ in range(self.args.K)]
        batch_bond_feat = [[] for _ in range(self.args.K)]
        batch_bond_neibor = [[] for _ in range(self.args.K)]
        query_idx, key_idx = [], []
        bond_batch_idx = [[] for _ in range(self.args.K)]
        batch_align_idx = [[], []]
        bias = 0
        bond_bias = [0 for _ in range(self.args.K)]

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
                for k in range(self.args.K):
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
        
        dist_block = self.args.dist_block
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
        return batch_data, raw_smi
    
    def predict(
        self,
        smi_dir: str,
        save_name: str,
        plot_num: Optional[int]=3,
        result_cano: Optional[bool]=True
    ):
        data, raw_smi = self.preprocess(smi_dir)
        with torch.no_grad():
            self.model.eval()
            data = data.pin_memory()
            beam_result, beam_scores = self.model.search(
                data=data.to(self.args.device),
                beam_size=self.args.beam_size,
                max_step=self.args.search_step,
                T=self.args.T,
                beam_group=self.args.beam_group,
                top_k=self.args.top_k,
                top_p=self.args.top_p
            )
            beam_result = beam_result.detach().cpu().numpy()
            beam_scores = beam_scores.detach().cpu().numpy()
        
        eos_ids = self.vocab['<EOS>']
        all_smi = []
        for batch_id, batch_res in enumerate(beam_result):
            beam_smi = []
            for beam_id, beam_res in enumerate(batch_res):
                res = beam_res
                if (res == eos_ids).sum() > 0:
                    res_eos = np.where(res == eos_ids)[0][0]
                    res = res[:res_eos]
                res_smi = [self.rvocab[idx] for idx in res]
                res_smi = ''.join(res_smi)
                if result_cano:
                    res_smi, valid = canonicalize_smiles(
                        smi=res_smi,
                        retain_smi=True,
                        map_clear=False,
                        cano_with_heavy_atom=False
                    )
                beam_smi.append(res_smi)
            beam_smi = ','.join(beam_smi)
            all_smi.append(beam_smi)

        root_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(root_dir, save_name)    
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'predict_result.txt'), 'w') as f:
            for i in range(len(raw_smi)):
                f.writelines('{id}.task:{task}\t input:{smi}\n'.format(id=i+1, task=self.eval_task, smi=raw_smi[i]))
                result = all_smi[i]
                result = result.split(',')
                scores = beam_scores[i]
                for j in range(len(result)):
                    f.writelines('{id}.:\t {score:.4}\t {smi}\n'.format(id=j+1, score=scores[j], smi=result[j]))
                f.writelines('\n')
        
        for i in range(len(raw_smi)):
            result = all_smi[i]
            result = result.split(',')
            result = [Chem.MolFromSmiles(smi) for smi in result]
            svg_draw = rdMolDraw2D.MolDraw2DSVG(300, 300)
            svg_draw.ClearDrawing()
            rdMolDraw2D.PrepareAndDrawMolecule(svg_draw, Chem.MolFromSmiles(raw_smi[i]))
            svg_draw.FinishDrawing()
            with open(os.path.join(save_dir, '{batch}-source.svg'.format(batch=i+1)), 'w') as f:
                f.write(svg_draw.GetDrawingText())
            for j in range(plot_num):
                svg_draw = rdMolDraw2D.MolDraw2DSVG(300, 300)
                svg_draw.ClearDrawing()
                rdMolDraw2D.PrepareAndDrawMolecule(svg_draw, result[j])
                svg_draw.FinishDrawing()
                with open(os.path.join(save_dir, '{batch}-{beam}.svg'.format(batch=i+1, beam=j+1)), 'w') as f:
                    f.write(svg_draw.GetDrawingText())
        
        print('visualize finish')

if __name__ == '__main__':
    parser = get_parser(mode='train')
    args = parser.parse_args()
    args.dataset_name = 'uspto_50k'
    args.vocab_name = 'uspto_50k'
    args.dropout = 0.0
    args.mode = 'eval'
    args.task = 'dualtask'
    args.eval_task = 'retrosynthesis'
    args.beam_size = 20
    args.T = 1.5
    args.norm_type = 'rmsnorm'
    args.ckpt_path = 'visualize_50k'
    args.ckpt_name = ['AVG_MAIN']
    args.device = 'cuda:0'
    setseed(args.seed)

    smi_dir = 'visualize_50k.txt'
    save_name = 'visualize_50k'

    smi = '[N+](=O)([O])c1cccc(CC=C)c1O.CC#CCBr'
    cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    onestep_predict = OneStepPrediction(args)
    onestep_predict.predict(
        smi_dir=smi_dir,
        save_name=save_name,
        plot_num=5,
        result_cano=False
    )