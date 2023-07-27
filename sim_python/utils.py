import torch

def tl_cdiv(num, deno):
    return num // deno if num % deno == 0 else num // deno + 1 

def tl_load(tensor, idx, mask=None, other=0.):
    if mask is not None:
        mask = mask.expand_as(idx)
        valid_idx = idx[mask]
        invalid_idx = idx[~mask]
    else:
        valid_idx = idx
    
    if valid_idx.max() >= tensor.numel():
        raise IndexError
 
    if mask is None or mask.all():
        output = tensor.flatten()[idx.flatten()].view_as(idx)
    else:
        output = torch.empty(size=idx.size(), dtype=tensor.dtype, device=tensor.device)
        output[mask] = tensor.flatten()[valid_idx.flatten()]
        output[~mask] = other
        output = output.view_as(idx)
    return output

def tl_store(container, idx, values, mask=None):
    if mask is None or mask.all():
        container.flatten()[idx.flatten()] = values.flatten()
    else:
        container.flatten()[idx[mask].flatten()] = values[mask].flatten() 
    return container
