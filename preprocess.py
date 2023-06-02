from parsers import get_parser
from model.model_utils import setseed
from model.preprocess.uspto_50k import uspto50k
from model.preprocess.uspto_MIT import usptoMIT
from model.preprocess.uspto_full import usptofull
from model.preprocess.tokenizer import Vocabulary, Tokenizer
from model.preprocess.featurize import Featurize
from model.preprocess.split_and_shuffle import Split_Shuffle


def preprocess(args):
    if args.smi2token:
        if args.dataset_name == 'uspto_50k': raw_process = uspto50k()
        elif args.dataset_name == 'uspto_MIT': raw_process = usptoMIT()
        elif args.dataset_name == 'uspto_full': raw_process = usptofull()
        raw_process.process()
    
    if args.generate_vocab:
        vocab_generate = Vocabulary()
        del vocab_generate
    
    if args.tokenize:
        tokenizer = Tokenizer(
            dataset_name=args.dataset_name,
            vocab_name=args.vocab_name
        )
        tokenizer.tokenize(augment_N=args.augment_N, max_perm_idx=args.max_perm_idx)
    
    if args.featurize:
        featurizer = Featurize(
            dataset_name=args.dataset_name,
            vocab_name=args.vocab_name
        )
        featurizer.featurize(
            max_split_count=args.file_split,
            max_deg=args.max_deg,
            K=args.K,
            kernel=args.kernel,
            max_bond_count=args.max_bond_count,
            max_dist_count=args.max_dist_count
        )
    
    if args.split_shuffle:
        eval_split = Split_Shuffle(
            dataset_name=args.dataset_name,
            vocab_name=args.vocab_name,
            K=args.K,
            kernel=args.kernel,
            seed=args.seed
        )
        eval_split.split_shuffle(args.split_data_len)

if __name__ == '__main__':
    parser = get_parser(mode='preprocess')
    args = parser.parse_args()
    setseed(args.seed)
    preprocess(args)