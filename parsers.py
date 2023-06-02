import argparse

from typing import Optional

def get_parser(
    mode: Optional[str]='train',
    parser: Optional[argparse.ArgumentParser]=None
):
    assert mode in ['train', 'eval', 'average', 'preprocess']
    if parser == None:
        parser = argparse.ArgumentParser(description=mode)
    parser = base_args(parser)
    if mode == 'preprocess':
        parser = preprocess_args(parser)
    elif mode in ['train']:
        parser = model_args(parser)
        parser = train_args(parser)
        parser = eval_args(parser)
    elif mode in ['eval']:
        parser = model_args(parser)
        parser = eval_args(parser)
    elif mode in ['average']:
        parser = average_args(parser)
    return parser

def base_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('base')
    group.add_argument('--model_name', help='model description name', type=str, default='GSA')
    group.add_argument('--dataset_name', help='dataset save name', type=str, choices=['uspto_50k', 'uspto_MIT', 'uspto_full'], default='uspto_50k')
    group.add_argument('--save_name', help='some description of save model', type=str, default='Model0')
    group.add_argument('--vocab_name', help='vocabulary name', type=str, default='all', choices=['all', 'uspto_50k', 'uspto_MIT', 'uspto_full'])
    group.add_argument('--seed', help='the random seed for model running', type=int, default=17)
    group.add_argument('--split_data_len', help='the data length after spliting eval data', type=int, default=0)
    group.add_argument('--augment_N', help='the (N-1) times of smiles will generate from augmentation', type=int, default=2)
    group.add_argument('--max_perm_idx', help='maximum atom index for SMILES random permute, the maximum value will be the padding', type=int, default=10)

    group.add_argument('--K', help='maximum hop for k-hop GNN', type=int, default=4)
    group.add_argument('--kernel', help='the kernel for calculating hop, (spd) for shortest path distance, (gd) for graph diffusion',\
                        type=str, default='spd', choices=['spd', 'gd'])
    group.add_argument('--max_bond_count', help='maximum count for each type of bond when computing peripheral subgraph', type=int, default=15)
    group.add_argument('--max_dist_count', help='maximum count for each distance when computing peripheral subgraph', type=int, default=15)
    group.add_argument('--max_deg', help='maximum degree to consider in attention bias', type=int, default=9)
    group.add_argument('--dist_block', help='the node distance block for embedding, if any distance not in this block,\
                                            it will be included into an extra embedding',
                        type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, [8, 15], [15, 2048]])

    group.add_argument('--beam_size', help='width for beam search', type=int, default=10)
    group.add_argument('--search_step', help='maximum step for beam search', type=int, default=300)
    group.add_argument('--T', help='temperature for beam search', type=float, default=1.0)
    group.add_argument('--beam_group', help='group for beam search', type=int, default=1)
    group.add_argument('--top_k', help='top_k filter for beam search sample', type=int, default=0)
    group.add_argument('--top_p', help='top_p filter for beam search sample', type=float, default=0.)
    
    return parser

def preprocess_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('preprocess')
    group.add_argument('--smi2token', help='preprocess csv tokenizer', action='store_true')
    group.add_argument('--generate_vocab', help='generate vocabulary', action='store_true')
    group.add_argument('--tokenize', help='preprocess token to matrix', action='store_true')
    group.add_argument('--featurize', help='preprocess token to graph', action='store_true')
    group.add_argument('--file_split', help='length for each preprocess file', type=int, default=10000)
    group.add_argument('--split_shuffle', help='generate splited data for evaling during training', action='store_true')

    return parser

def model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model')
    group.add_argument('--batch_size', help='the product molecule graph num for each step', type=int, default=64)
    group.add_argument('--token_limit', help='the maximun token number of product+reactant for each step', type=int, default=0)
    group.add_argument('--d_model', help='hidden size of model', type=int, default=256)
    group.add_argument('--d_ff', help='hidden size of feed-forward-network', type=int, default=256 * 8)
    group.add_argument('--enc_head', help='attention head of encoder attention', type=int, default=8)
    group.add_argument('--dec_head', help='attention head of decoder attention', type=int, default=8)
    group.add_argument('--enc_layer', help='layer num of encoder', type=int, default=6)
    group.add_argument('--dec_layer', help='layer num of decoder', type=int, default=8)
    group.add_argument('--dropout', help='dropout rate of model', type=float, default=0.3)
    group.add_argument('--mode', help='training or evaling', type=str, default='train', choices=['train', 'eval'])
    group.add_argument('--task', help='model task', type=str, default='dualtask', choices=['dualtask', 'retrosynthesis', 'forwardsynthesis'])
    group.add_argument('--reaction_class', help='use reaction class or not', action='store_true', default=False)
    group.add_argument('--pe_type', help='positional embedding type for decoder', type=str, default='rope', choices=['rope', 'abs'])
    group.add_argument('--ffn_type', help='feedforward type for transformer', type=str, default='vanilla', choices=['glu', 'vanilla'])
    group.add_argument('--norm_type', help='normalization type for model', type=str, default='rmsnorm', choices=['rmsnorm', 'layernorm'])
    group.add_argument('--device', help='the device for model running', type=str, default='cuda:0')

    return parser

def train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('train')
    group.add_argument('--train_eval', help='if True, module will evaling during training', action='store_true', default=True)
    group.add_argument('--eval_batch_size', help='the product molecule graph num for each step(eval)', type=int, default=64)
    group.add_argument('--eval_token_limit', help='the maximun token number of product+reactant for each step(eval)', type=int, default=0)
    group.add_argument('--eval_start', help='the epoch to start evaling', type=int, default=100000)
    group.add_argument('--eval_step', help='the step for each evaling', type=int, default=2000)
    group.add_argument('--epochs', help='the total epochs to finish training', type=int, default=1000)
    group.add_argument('--steps', help='the total steps to finish training', type=int, default=150000)
    group.add_argument('--memory_clear_count', help='pytorch memory clear count in each epoch', type=int, default=0)
    group.add_argument('--accum_count', help='the gradient update accum count', type=int, default=2)
    group.add_argument('--optimizer', help='the optimizer name for training', type=str, choices=['AdamW'], default='AdamW')
    group.add_argument('--lrschedule', help='the lrschedule name for training', type=str, choices=['rsqrt', 'cosine'], default='rsqrt')
    group.add_argument('--lr_factor', help='the basic learning rate scale for lr_schedule(in original, it will be a scale rate for real lr)', type=float, default=1.0)
    group.add_argument('--max_lr', help='maximum learning rate for cosine learning rate', type=float, default=3e-4)
    group.add_argument('--min_lr', help='minimum learning rate for cosine learning rate', type=float, default=1e-6)
    group.add_argument('--warmup', help='the step to reach the maximum learning rate', type=int, default=10000)
    group.add_argument('--end_step', help='the step to reach the minimum learning rate', type=int, default=150000)
    group.add_argument('--labelsmooth', help='the prob for negative label loss calculation', type=float, default=0.1)
    group.add_argument('--gamma', help='the scale number of difficulty weight according to the probability of each token in a minibatch',
                       type=float, default=2.0)
    
    return parser

def eval_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('eval')
    group.add_argument('--eval_task', help='model task for evaling', type=str, default='retrosynthesis', choices=['retrosynthesis', 'forwardsynthesis'])
    group.add_argument('--eval_mode', help='use eval data or test data', type=str, default='test', choices=['eval', 'test'])
    group.add_argument('--ckpt_path', help='folder name of checkpoints', type=str)
    group.add_argument('--ckpt_name', help='checkpoints for evaling', nargs='+', type=str, default=[])
    
    return parser

def average_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('average')
    group.add_argument('--ckpt_path', help='folder name of checkpoints', type=str)
    group.add_argument('--average_list', help='use these checkpoint to generate an average model, it will have much better performance usually(especially in top3-10)',
                       nargs='+', type=str, default=[])
    group.add_argument('--average_name', help='the save name of this average checkpoint', type=str, default='AVG_MAIN')

    return parser