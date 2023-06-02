import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional
from torch.optim import AdamW
from tqdm.std import trange
from torch.utils.data import DataLoader
from parsers import get_parser
from model.batch_loader import ReactionDataset
from model.prediction_model import SMILEdit
from model.model_utils import CKPT_PATH, setseed, search_result_process, RsqrtLearningRate, CosineLearningRate, ModelSave

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def optimizer_select(
    name: str,
    param:nn.Parameter,
    eps: Optional[float]=1e-5,
    weight_decay: Optional[float]=0.
):
    if name == 'AdamW':
        return AdamW(
            params=param,
            lr=1.0,
            eps=eps,
            weight_decay=0.
        )

def lrschedule_select(
    name: str,
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup: Optional[int]=10000,
    lr_factor: Optional[float]=1.0,
    max_lr: Optional[float]=3e-4,
    min_lr: Optional[float]=1e-5,
    end_step: Optional[int]=150000,
):
    if name == 'rsqrt':
        return RsqrtLearningRate(
            optimizer=optimizer,
            d_model=d_model,
            warmup=warmup,
            lr_factor=lr_factor
        )
    elif name == 'cosine':
        return CosineLearningRate(
            optimizer=optimizer,
            warmup=warmup,
            max_lr=max_lr,
            min_lr=min_lr,
            end_step=end_step
        )

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def evaling(
    args,
    model_save: ModelSave,
    model: SMILEdit,
    eval_dataset: ReactionDataset,
    vocab: dict[str:int],
    rvocab: dict[int:str],
    ckpt_name: str|int,
    logger: Optional[logging.Logger]=None,
    writer: Optional[SummaryWriter]=None
):
    seq_acc = np.zeros((10))
    seq_invalid = np.zeros((10))
    predictions = []
    if logger is not None:
        logger.info(f'step {ckpt_name} eval start')

    with torch.no_grad():
        model.eval()
        loader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=eval_dataset.process_data,
            num_workers=0,
            drop_last=False,
            shuffle=False
        )
        tqdm_step = trange(len(loader))
        for step, data in zip(tqdm_step, loader):
            data = data.pin_memory()
            search_result, search_scores = model.search(
                data=data.to(args.device),
                beam_size=args.beam_size,
                max_step=args.search_step,
                T=args.T,
                beam_group=args.beam_group,
                top_k=args.top_k,
                top_p=args.top_p
            )
            beam_acc, beam_invalid, beam_smi = search_result_process(
                tgt_seq=data.tgt,
                vocab=vocab,
                rvocab=rvocab,
                beam_result=search_result
            )
            seq_acc += beam_acc
            seq_invalid += beam_invalid
            predictions.extend(beam_smi)
    
    seq_acc = np.cumsum(seq_acc) / len(eval_dataset)
    seq_invalid = np.cumsum(seq_invalid) / np.array([i * len(eval_dataset) for i in range(1, 10 + 1, 1)])
    if writer is not None:
        writer.add_scalars(main_tag='Acc/eval', tag_scalar_dict={'t1':seq_acc[0], 't3':seq_acc[2], 't5':seq_acc[4], 't10':seq_acc[9], 'wt':model_save.weight_acc(seq_acc)}, global_step=ckpt_name)
        writer.add_scalars(main_tag='Invalid/eval', tag_scalar_dict={'v1':seq_invalid[0], 'v3':seq_invalid[2], 'v5':seq_invalid[4], 'v10':seq_invalid[9]}, global_step=ckpt_name)
    if logger is not None:
        logger.info('step {step} eval finish, reaction count {count}, top1,3,5,10 acc is [{t1:.4}, {t3:.4}, {t5:.4}, {t10:.4}]'\
                    .format(step=ckpt_name, count=len(eval_dataset), t1=seq_acc[0], t3=seq_acc[2], t5=seq_acc[4], t10=seq_acc[9]))
    
    if args.mode == 'train':
        model_save.save(
            model=model,
            step=ckpt_name,
            acc=seq_acc
        )
    model_save.eval_record(
        mode='train' if args.mode == 'train' else args.eval_mode,
        model_name=ckpt_name,
        seq_acc=seq_acc,
        seq_invalid=seq_invalid,
        beam_size=args.beam_size,
        T=args.T
    )
    if args.mode == 'train':
        model.train()
        return model_save, eval_dataset


def training(args):
    train_dataset = ReactionDataset(
        dataset_name=args.dataset_name,
        vocab_name=args.vocab_name,
        mode='train',
        task=args.task,
        reaction_class=args.reaction_class,
        token_limit=args.token_limit,
        K=args.K,
        kernel=args.kernel,
        dist_block=args.dist_block,
        split_data_len=0,
        shuffle=True,
        augment_N=args.augment_N,
        max_perm_idx=args.max_perm_idx
    )
    vocab = train_dataset.vocabulary
    rvocab = {v:k for k, v in vocab.items()}
    if args.train_eval:
        eval_dataset = ReactionDataset(
            dataset_name=args.dataset_name,
            vocab_name=args.vocab_name,
            mode='eval',
            task=args.eval_task,
            reaction_class=args.reaction_class,
            token_limit=args.eval_token_limit,
            K=args.K,
            kernel=args.kernel,
            dist_block=args.dist_block,
            split_data_len=args.split_data_len,
            shuffle=False,
            augment_N=1,
            max_perm_idx=args.max_perm_idx
        )

    start_time = time.strftime('%y%m%d %H%M', time.localtime(time.time()))
    ckpt_dir = os.path.join(CKPT_PATH, args.save_name)
    model_dir = os.path.join(ckpt_dir, start_time)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    logger = logging.getLogger()
    log_file = logging.FileHandler(os.path.join(model_dir, 'log.log'))
    log_console = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s %(message)s')
    log_file.setFormatter(log_format)
    log_console.setFormatter(log_format)
    logger.addHandler(log_file)
    logger.addHandler(log_console)
    logger.setLevel('INFO')

    for k, v in args.__dict__.items():
        logger.info('args -> {0}: {1}'.format(k, v))
    
    writer = SummaryWriter(
        log_dir=os.path.join(model_dir, 'tblog')
        # log_dir=f'/root/tf-logs/{start_time}' # for autodl
    )
    
    model = SMILEdit(
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
        vocab=vocab,
        task=args.task,
        reaction_class=args.reaction_class,
        pe_type=args.pe_type,
        ffn_type=args.ffn_type,
        norm_type=args.norm_type,
        labelsmooth=args.labelsmooth,
        gamma=args.gamma,
        augment_N=args.augment_N,
        max_perm_idx=args.max_perm_idx,
        device=args.device
    )
    optimizer = optimizer_select(
        name=args.optimizer,
        param=model.parameters(),
        eps=1e-5,
        weight_decay=0.01
    )
    schedule = lrschedule_select(
        name=args.lrschedule,
        optimizer=optimizer,
        d_model=args.d_model,
        warmup=args.warmup,
        lr_factor=args.lr_factor,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        end_step=args.end_step
    )
    model_save = ModelSave(
        ckpt_dir=model_dir,
        const_save=[],
        w1=0.9
    )

    logger.info(model)
    logger.info(f'parameters: {sum(_.numel() for _ in model.parameters())}')
    
    #----------------------------------start training----------------------------
    stepcount = 0
    accumcount = 0
    eval_step = []
    if args.train_eval:
        eval_step = [_ for _ in range(args.eval_start, args.steps, args.eval_step)]

    model = model.to(args.device)
    # model = torch.compile(model) #torch.complie BETA，run it on Linux
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    logging.info('start training')

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=train_dataset.process_data,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )
        tqdm_step = trange(len(loader))
        clear_step = [(len(loader) // (args.memory_clear_count + 1)) * i for i in range(1, args.memory_clear_count + 1, 1)]
        epoch_loss = 0.
        step_loss = 0.
        epoch_seq_acc = [0., 0.]
        epoch_token_acc = [0., 0.]
        step_seq_acc = [0., 0.]
        step_token_acc = [0., 0.] #retrosynthesis, forward synthesis

        for step, data in zip(tqdm_step, loader):
            if stepcount == args.steps: break
            if step in clear_step: torch.cuda.empty_cache()
            data = data.pin_memory()

            with torch.cuda.amp.autocast(enabled=True):
                loss, retro_acc, forward_acc = model(data.to(args.device), writer)
            loss = loss / args.accum_count
            scaler.scale(loss).backward()

            epoch_loss += loss.item() * args.accum_count
            step_loss += loss.item()
            epoch_seq_acc[0] += retro_acc['seq']
            epoch_seq_acc[1] += forward_acc['seq']
            epoch_token_acc[0] += retro_acc['token']
            epoch_token_acc[1] += forward_acc['token']
            step_seq_acc[0] += retro_acc['seq'] / args.accum_count
            step_seq_acc[1] += forward_acc['seq'] / args.accum_count
            step_token_acc[0] += retro_acc['token'] / args.accum_count
            step_token_acc[1] += forward_acc['token'] / args.accum_count
            # writer.add_scalar(tag='LR/train', scalar_value=get_lr(optimizer), global_step=schedule._step_count)

            accumcount += 1
            if accumcount == args.accum_count:
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                if not (scale > scaler.get_scale()):
                    schedule.step()
                optimizer.zero_grad()
                accumcount = 0
                stepcount += 1
            
            if accumcount == 0:
                tqdm_step.set_description('epoch {epoch}, step: {step}, lr: {lr:.4} loss: {loss:.6}, ret[{ret_t:.4}, {ret_s:.4}], fwd[{fwd_t:.4}, {fwd_s:.4}]'.format(
                    epoch=epoch, step=stepcount, lr=get_lr(optimizer), loss=step_loss, ret_t=step_token_acc[0], ret_s=step_seq_acc[0], fwd_t=step_token_acc[1],
                    fwd_s=step_seq_acc[1]
                ))
                writer.add_scalar(tag='Loss(step)/train', scalar_value=step_loss, global_step=stepcount)
                writer.add_scalars(main_tag='Acc(step)/train', tag_scalar_dict={'ret_t':step_token_acc[0], 'ret_s':step_seq_acc[0], 'fwd_t':step_token_acc[1], 'fwd_s':step_seq_acc[1]}, global_step=stepcount)
                step_loss = 0
                step_token_acc = [0., 0.]
                step_seq_acc = [0., 0.]
            
                if stepcount in eval_step:
                    model_save, eval_dataset = evaling(args, model_save, model, eval_dataset, vocab, rvocab, stepcount, logger, writer)
        
        if stepcount == args.steps: break
        if args.accum_count > 1 and accumcount > 0:
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            if not (scale > scaler.get_scale()):
                schedule.step()
            optimizer.zero_grad()
            accumcount = 0
            stepcount += 1
            tqdm_step.set_description('epoch {epoch}, step: {step}, lr: {lr:.4} loss: {loss:.6}, ret[{ret_t:.4}, {ret_s:.4}], fwd[{fwd_t:.4}, {fwd_s:.4}]'.format(
                epoch=epoch, step=stepcount, lr=get_lr(optimizer), loss=step_loss, ret_t=step_token_acc[0], ret_s=step_seq_acc[0], fwd_t=step_token_acc[1],
                fwd_s=step_seq_acc[1]
            ))
            step_loss = 0
            step_token_acc = [0., 0.]
            step_seq_acc = [0., 0.]

            if stepcount in eval_step:
                model_save, eval_dataset = evaling(args, model_save, model, eval_dataset, vocab, rvocab, stepcount, logger, writer)
        
        epoch_loss = epoch_loss / len(loader)
        epoch_seq_acc[0] = epoch_seq_acc[0] / len(loader)
        epoch_seq_acc[1] = epoch_seq_acc[1] / len(loader)
        epoch_token_acc[0] = epoch_token_acc[0] / len(loader)
        epoch_token_acc[1] = epoch_token_acc[1] / len(loader)
        writer.add_scalar(tag='Loss(epoch)/train', scalar_value=epoch_loss, global_step=epoch)
        writer.add_scalars(main_tag='Acc(epoch)/train', tag_scalar_dict={'ret_t':epoch_token_acc[0], 'ret_s':epoch_seq_acc[0], 'fwd_t':epoch_token_acc[1], 'fwd_s':epoch_seq_acc[1]}, global_step=epoch)

        logger.info('--------> epoch {epoch}, step_sum: {step}, loss: {loss:.6}, ret[{ret_t:.4}, {ret_s:.4}], fwd[{fwd_t:.4}, {fwd_s:.4}]'.format(
            epoch=epoch, step=stepcount, loss=epoch_loss, ret_t=epoch_token_acc[0], ret_s=epoch_seq_acc[0], fwd_t=epoch_token_acc[1],
            fwd_s=epoch_seq_acc[1]
        ))  
    
    if stepcount not in eval_step:
        model_save, eval_dataset = evaling(args, model_save, model, eval_dataset, vocab, rvocab, stepcount, logger, writer)
    logger.info('training finish')
    writer.close()

if __name__ == '__main__':
    parser = get_parser(mode='train')
    args = parser.parse_args()
    setseed(args.seed)

    if args.mode == 'train':
        training(args)
    elif args.mode == 'eval':
        # args.ckpt_path = "" # enter your ckpt path
        # args.ckpt_name = ['AVG_MAIN'] # enter your ckpt name

        eval_dataset = ReactionDataset(
            dataset_name=args.dataset_name,
            vocab_name=args.vocab_name,
            mode=args.eval_mode,
            task=args.eval_task,
            reaction_class=args.reaction_class,
            token_limit=args.eval_token_limit,
            K=args.K,
            kernel=args.kernel,
            dist_block=args.dist_block,
            split_data_len=args.split_data_len,
            shuffle=False,
            augment_N=1,
            max_perm_idx=args.max_perm_idx
        )
        ckpt_dir = os.path.join(CKPT_PATH, args.save_name)
        ckpt_dir = os.path.join(ckpt_dir, args.ckpt_path)
        vocab = eval_dataset.vocabulary
        rvocab = {v:k for k, v in vocab.items()}
        model_save = ModelSave(
            ckpt_dir=ckpt_dir,
            const_save=[],
            w1=0.9
        )
        model = SMILEdit(
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
            vocab=vocab,
            task=args.task,
            reaction_class=args.reaction_class,
            pe_type=args.pe_type,
            ffn_type=args.ffn_type,
            norm_type=args.norm_type,
            augment_N=args.augment_N,
            max_perm_idx=args.max_perm_idx,
            device=args.device
        )
        for ckpt in args.ckpt_name:
            model = model_save.load(
                model_name=ckpt,
                model=model,
                device=args.device
            )
            evaling(args, model_save, model.to(args.device), eval_dataset, vocab, rvocab, ckpt)