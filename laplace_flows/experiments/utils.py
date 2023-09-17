from copy import deepcopy
import numpy as np
from typing import Any, Iterable, Union, Tuple, List, Optional, Dict
from pathlib import Path
from importlib import import_module
import yaml

import torch

from ray import tune

def split_data(
        data: np._typing.ArrayLike,
        split: Iterable[float], 
        return_shuffle=True
    ) -> Union[Iterable[np._typing.ArrayLike], Tuple[Iterable[np._typing.ArrayLike], List[np.array]]]:
    """Generates a random partition of the data according to the given split $(p_1,\ldots,p_N)$.

    @param data: The dataset.
    @param split: A sequence of $N$ real numbers $p_i$, $0 \leq p_i \leq 1$ such that $\sum_{i=1}^N p_i \leq 1$. 
    In case of $\sum_{i=1}^N < 1$, the function computes an $N+1$ split with the last component implicitly given.
    @returns: If $\sum_{i=1}^N = 1$ then the function return an random $N$-partition as tuple where the $i$th 
    component contains a $p_i$th fraction of the data (deviation due to rounding errors possible). 
    If $\sum_{i=1}^N < 1$, then the function returns
    $\textbf{split_data}(data, [p_1,\ldots,p_n,1-\sum_i p_i])$.
    """

    # Consistency checks
    if any(x < 0 for x in split):
        raise ValueError("All split components must be positive!")
    checksum = sum(split)
    if checksum == 1:
        split = split[:-1]
    elif checksum > 1:
        raise ValueError("The sum of all split components must not be larger than 1!")
    

    N = data.shape[0]
    idxs = []
    partial_sum = 0
    for p in split:
        fraction = int(p * N)
        idxs.append(partial_sum + fraction)
        partial_sum += fraction
    idxs.append(N)

    shuffle = np.random.choice(N, N, replace=False)
    data  = data[shuffle]

    partition = []
    partition_shuffle = []
    for start, end in zip([0] + idxs, idxs):
        partition.append(data[start: end])
        partition_shuffle.append(shuffle[start: end])
    
    return (partition, partition_shuffle) if return_shuffle else partition
        


def config_from_yaml(yaml_path: Union[str, Path]) -> dict:
    """Loads a yaml file and returns the corresponding dictionary.
    Besides the standard yaml syntax, the function also supports the following
    additional functionality:
    
    Special keys:
    __class__<key>: The value of this key is interpreted as the class name of the object. 
    The class is imported and stored in the result dictionary under the key <key>.
    Example:
        entry in yaml: __class__model: laplace_flows.flows.NiceFlow)
        entry in result: model: __import__("laplace_flows.flows.NiceFlow")
    __tune__<key>: The value of this key is interpreted as a dictionary that contains the 
    configuration for the hyperparameter optimization using tune sample methods. 
    the directive is evaluated and the result in the result dictionary under the key <key>.
    Example:
        entry in yaml: __tune__lr: loguniform(1e-4, 1e-1)
        entry in result: lr: eval("tune.loguniform(1e-4, 1e-1)")

    Args:
        yaml_path: Path to the yaml file.
    """
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        
    def parse(d: dict):
        result = dict()
        for k, v in d.items():
            if isinstance(k, str):
                if k.startswith("__class__"):
                    module, cls = v.rsplit(".", 1)
                    result[k[9:]] = getattr(import_module(module), cls)
                elif k.startswith("__tune__"):
                    import torch
                    result[k[8:]] = eval(f"tune.{str(v)}")
                elif isinstance(v, dict):
                    result[k] = parse(v)
                elif isinstance(v, list):
                    result[k] = [parse_dict(x) for x in v]
                else:
                    result[k] = v
            else:
                result[k] = v


        return result
    
    config = parse(config)
    return config

def create_checkerboard_mask(h, w, invert=False):
    """Creates a checkerboard mask of size $(h,w)$.

    Args:
        h (_type_): height
        w (_type_): width
        invert (bool, optional): If True, inverts the mask. Defaults to False.

    Returns:
        _type_: _description_
    """
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def unfold_raw_config(d: Dict[str, Any]):
    """Unfolds an ordered DAG given as a dictionary into a tree given as dictionary. 
    That means that unfold_dict(d) is bisimilar to d but no two distinct key paths in the resulting
    dictionary reference the same object
    
    Args:
        d: The dictionary to unfold
    """
    du = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            du[k] = unfold_raw_config(v)
        elif isinstance(v, list):
            du[k] = [unfold_raw_config(x) for x in v]
        else:
            du[k] = deepcopy(v)
    
    return du

def push_overwrites(item: Any, attributes: Dict[str, Any]) -> Any:
    """Pushes the overwrites in the given dictionary to the given item.
    
    If the item already specifies an overwrite, it is updated.
    If the item is a dictionary, an overwrite specification for the dictionary is created. 
    If the item is a list, the overwrites are pushed each element and the processed list is returned.
    Otherwise, item is overwritten by attributen
    
    Args:
        item: The item to push the overwrites to.
        overwrites: The overwrites to push.
    """
    try:
        if "__exact__" in attributes:
                return deepcopy(attributes["__exact__"])
    except:
        pass
    
    if isinstance(item, dict):
        if "__overwrites__" not in item:
            result = deepcopy(attributes)
            result["__overwrites__"] = item
        else:
            result = item
            result["__overwrites__"].update(attributes)
    elif isinstance(item, list): 
        result = [push_overwrites(x, attributes) for x in item]
    else:
        result = deepcopy(attributes)
        
    return result
    

def apply_overwrite(d: Dict[str, Any], recurse=True):
    """Applies the "__overwrites__" keyword sematic to a unfolded raw config dictionary and returns the result.
    Except for the special semantics that applies to dictionaries and lists (see below),
    all keys $k$ that are present in in the  "__overwrites__" dictionary $o$ are overwritten in $d$ by $o[k]$. 
    
    ** Dict/List overwrites **:
    - If $d[k]$ is a dictionary, then $d[k]$ must be a dictionary and overwrites of $o[k]$ are recursively 
    to the corresponding $d[k]$, i.e. lower-lever overwrites are specified/updated (see notes on recursion).
    The only exception is if $o[k]$ contains the special key "__exact__" with value True. In this case
    $d[k$]$ is replaced by $o[k]["__exact__"]$.
    - If $o[k]$ is a list, then $o[k]$ is pushed to all list elements. 
    
    ** Recursion **:
    If recursion is enabled, overwrites are are fully expanded in infix order where nested overwrites 
    (see behavior on dict/list overwrites) are pushed (and overwrite) to the next level, i.e. 
    higher level overwrites lower level. Else, only the top-level overwrites are applied.
    
    ** Note **: Applying this function to a non-unfolded dictionary
    can result in unexpected behavior due to side side-effects.
      
    Args:
        d: The unfolded raw config dictionary.
        recurse: If True, the overwrites are applied recursively. Defaults to True. 
            Can be useful for efficient combination of this method with other parsing methods.
    """
    
    # Apply top-level overwrite
    if "__overwrites__" in d:
        overwritten_attr = d 
        d = overwritten_attr.pop("__overwrites__")
        
        for k, v in overwritten_attr.items():
            if k not in d:
                d[k] = v
            else:
                d[k] = push_overwrites(d[k], v)
            
                    
    if recurse:
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = apply_overwrite(v)
            elif isinstance(v, list):
                d[k] = [apply_overwrite(x) for x in v]
            
    return d
    
        
def read_config(yaml_path: Union[str, Path]) -> dict:
    """Loads a yaml file and returns the corresponding dictionary.
    Besides the standard yaml syntax, the function also supports the following
    additional functionality:
    
    Special keys:
    __class__: The value of this key is interpreted as the class name of the object. 
    The class is imported and stored in the result dictionary under the key <key>.
    Example:
        entry in yaml: __class__model: laplace_flows.flows.NiceFlow)
        entry in result: model: __import__("laplace_flows.flows.NiceFlow")
    __tune__<key>: The value of this key is interpreted as a dictionary that contains the 
    configuration for the hyperparameter optimization using tune sample methods. 
    the directive is evaluated and the result in the result dictionary under the key <key>.
    Example:
        entry in yaml: __tune__lr: loguniform(1e-4, 1e-1)
        entry in result: lr: eval("tune.loguniform(1e-4, 1e-1)")

    Args:
        yaml_path: Path to the yaml file.
    """
    
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = unfold_raw_config(config)   
    config = parse_raw_config(config)

    return config
        
def parse_raw_config(d: dict):
    """Parses an unfolded raw config dictionary and returns the corresponding dictionary.
    Parsing includes the following steps:
    - Overwrites are applied (see apply_overwrite)
    - The "__object__" key is interpreted as a class name and the corresponding class is imported.
    - The "__eval__" key is evaluated.
    - The "__class__" key is interpreted as a class name and the corresponding class is imported.
    
    Args:
        d: The raw config dictionary.
    """
    if isinstance(d, dict):
        d = apply_overwrite(d, recurse=False)
            
        # Depth-first recursion
        for k, v in d.items():
            d[k] = parse_raw_config(v)
        
        if "__object__" in d:
            module, cls = d["__object__"].rsplit(".", 1)
            C = getattr(import_module(module), cls)
            d.pop("__object__")
            return C(**d)
        elif "__eval__" in d:
            return eval(d["__eval__"])
        elif "__class__" in d:
            module, cls = d["__class__"].rsplit(".", 1)
            C = getattr(import_module(module), cls)
            return C
        else:    
            return d
    elif isinstance(d, list):
        result = [parse_raw_config(x) for x in d]
        return result
    else:
        return d
            

