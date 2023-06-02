import torch
import IndexElemtwiseCUDA

from typing import Optional

OPERATE_VAL = {"sum", "mul"}

def broadcast(
    src1: torch.Tensor,
    src2: torch.Tensor
) -> list[torch.Tensor]:
    if src1.dim() != src2.dim():
        raise ValueError("the diminsion of inputs tenosor must be the same")
    
    expand_size1, expand_size2 = list(src1.size()), list(src2.size())
    for _ in range(1, src1.dim()):
        dim_size = expand_size1[_] if expand_size1[_] > expand_size2[_] else expand_size2[_]
        expand_size1[_] = dim_size
        expand_size2[_] = dim_size
    
    return src1.expand(expand_size1).contiguous(), src2.expand(expand_size2).contiguous()

def idx_check(
    src: torch.Tensor,
    idx: torch.Tensor | None
) -> torch.Tensor:
    if idx is None:
        return torch.arange(src.size(0), device=src.device)
    else:
        return idx

def operate_check(operate_name: Optional[str]=None) -> str:
    if operate_name is None: operate_name = "mul"
    if operate_name not in OPERATE_VAL:
        raise ValueError(f"operate \'{operate_name}\' not in the operate list")
    return operate_name

def indexelemwise(
    src1: torch.Tensor, src2: torch.Tensor,
    src_idx1: Optional[torch.Tensor]=None,
    src_idx2: Optional[torch.Tensor]=None,
    operate: Optional[str]=None
) -> torch.Tensor:
    """
    the custom version for operation 'src1.index_select(0, idx1) ~ src2.index_select(0, idx2)'\\
    operate-> sum / mul(default)\\
    idx2 is optional for 'src1.index_select(0, idx1) ~ src2'
    """
    src1, src2 = broadcast(src1, src2)
    src_idx1 = idx_check(src1, src_idx1)
    operate = operate_check(operate)
    return IndexElemtwiseCUDA.IndexElemtwiseOperate(src1, src2, src_idx1, src_idx2, operate)