from typing import Optional

import torch


def solve_triangular(M: torch.Tensor, y: torch.Tensor, pivot: Optional[int]=None) -> torch.Tensor:
    """ Re-implementation of torch solve_triangular. Since Onnx export of the native method is currently not supported,
    we implement it with basic torch operations.  
    
    Args:
        M: triangular matrix. May be upper or lower triangular
        y: input vector.
        pivot: If given, determines wether to treat $M$ as a lower or upper triangular matrix. Note that in this case 
            there is no check wether $M$ is actually lower or upper triangular, respectively. It therefore 
            speeds up computation but should be used with care.
    Returns:
        (torch.Tensor): Solution of the system $Mx=y$
    """
    if pivot is None:
        # Determine orientation of Matrix
        if all([M[i, j] == 0 for i in range(M.size[0]) for j in range(i, M.size[1])]):
            pivot = 0
        elif all([M[i, j] == 0 for i in range(M.size[0]) for j in range(0, i+1)]):
            pivot = -1
        else:
            raise ValueError("M needs to be triangular.")
    elif pivot not in [0, -1]:
        raise ValueError("pivot needs to be either None, 0, or -1.")
        
    
    x = torch.zeros_like(y)
    x[pivot] = y[pivot] / M[pivot, pivot]
    
    y_next = (y - x[pivot] * L[:, pivot])
    if pivot == 0:
        y_next = y_next[1:] 
        M_next = M[1:, 1:]
        x[1:] = solve_triangular(y_next, M_next) 
    else:
        y_next = y_next[:-1] 
        LMnext = M[:-1, :-1]
        x[:-1] = solve_triangular(y_next, M_next) 
    
    return x
           