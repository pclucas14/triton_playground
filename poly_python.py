import torch
DTYPE = torch.float32
# --- ARGUMENTS

# Pointers to matrices
mixing_weights_ptr, weights_ptr, input_ptr, output_ptr = 0, 0, 0, 0

# Matrix dimensions
bs, n_skills, rank, d_in, d_out = 16, 6, 12, 129, 256

# generate weights a la polytropon
module_logits = torch.randn(bs, n_skills, dtype=DTYPE, device='cuda')
mixing_weights = torch.nn.functional.softmax(module_logits, dim=-1)
skill_weights = torch.randn(n_skills, d_in, rank, dtype=DTYPE, device='cuda')
input_ = torch.randn(bs, d_in, dtype=DTYPE, device='cuda')
output_ = torch.empty(bs, rank, dtype=DTYPE, device='cuda')

# meta parameters
block_size_m = 5 
block_size_n = 7
block_size_k = 3
group_size_m = 4

A = torch.einsum("bs,sdr->bdr", (mixing_weights, skill_weights))
A = A.reshape(bs, d_in, rank)
exp_out = torch.einsum('bi,bio->bo', (input_, A))

def ceil_div(num, deno):
    return num // deno if num % deno == 0 else num // deno + 1 

use_groups = True

M, N = output_.size()
K = d_in

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

# simulate launch grid
for pid in range(ceil_div(M, block_size_m) * ceil_div(N, block_size_n)):

    # simulate a kernel call
    if use_groups:
        # get better l2-cache utilization by grouping
        num_pid_m = ceil_div(M, block_size_m)
        num_pid_n = ceil_div(N, block_size_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        print(f'pid_m: {pid_m}, pid_n: {pid_n}')
        pass
    else:
        # each pid computes one (i,j) / (m, n) entry in the final matrix
        num_blocks_n = ceil_div(N, block_size_n)
        pid_m = pid // num_blocks_n
        pid_n = pid % num_blocks_n

        print(f'pid_m: {pid_m}, pid_n: {pid_n}')

    # compute the offsets into the input matrices
    # this is exactly correct if strides for this dimension are 1. 
    offset_am =  pid_m * block_size_m + torch.arange(block_size_m) # this is the input. So ok 
    offset_bn =  pid_n * block_size_n + torch.arange(block_size_n) # this is the weights
    offset_k = torch.arange(block_size_k)

    # account for strides
    # (block_size_m, block_size_k) [These are 2-D indices for X
    # we will progressively move the k index to get the full row
    # this is the input. so ok
    X_ptr = input_ptr + input_.stride(0) * offset_am[:, None] + input_.stride(1) * offset_k[None, :]

    # this is the weight tensor
    offset_skills = torch.arange(n_skills)
    Y_ptr = weights_ptr + skill_weights.stride(2) * offset_bn[None, None, :] + \
                          skill_weights.stride(1) * offset_k[None, :, None] + \
                          skill_weights.stride(0) * offset_skills[:, None, None]

    # additionally, we have to index the mixing_coefs. Here the batch size is the first (index 0) dim.
    offset_w = pid_m * block_size_m + torch.arange(n_skills)
    offset_w = torch.arange(n_skills)
    mix_ptr = mixing_weights_ptr + mixing_weights.stride(0) * offset_am[:, None] + mixing_weights.stride(1) * offset_w[None, :]

    accumulator = torch.zeros((block_size_m, block_size_n), dtype=input_.dtype, device=input_.device)
    # for k_chunk in range(0, K, block_size_k):
    for k_chunk in range(0, ceil_div(K, block_size_k)): # makes the masking of invalid outputs easier

        # simulate 1d indexing as in triton                               # actual shapes          # dimensions being indexed
        try:
            x_mask   = (offset_am[:, None]  < M) & (offset_k[None, :] < K - k_chunk * block_size_k)
            ws_mask  = (offset_k[None, :, None] < K - k_chunk * block_size_k) & (offset_bn[None, None, :] < N)
            mw_mask  = (offset_am[:, None]  < M)  
            x_chunk  = tl_load(input_, X_ptr, mask=x_mask)            # BSM, BSK               # bs,     d_in         (M, K)
            ws_chunk = tl_load(skill_weights, Y_ptr, mask=ws_mask)    # n_skills, BSN, BSK     # skills, d_in, d_out  (_, K, N)
            mw_chunk = tl_load(mixing_weights, mix_ptr, mask=mw_mask) # BSM, n_skills          # bs,     skills       (M, _)
        except IndexError as e:
            breakpoint()
            xx = 1

        # let's just iterate over the skills for now
        for skill_idx in range(n_skills):
            accumulator += mw_chunk[:, skill_idx, None] * torch.matmul(x_chunk, ws_chunk[skill_idx])

        # move pointers
        X_ptr += block_size_k * input_.stride(1)
        Y_ptr += block_size_k * skill_weights.stride(1)

    # expected_tmp = torch.zeros_like(accumulator)
    # x_chunk = input_[offset_am]
    # A_chunk = torch.einsum("bs,sdr->bdr", (mixing_weights[offset_am], skill_weights[:, :, offset_bn]))
    # out = torch.bmm(x_chunk.unsqueeze(1), A_chunk).squeeze()
    # breakpoint()

    # compute the offsets into the output matrix (think of this as 1D-indexing)
    offset_om = pid_m * block_size_m + torch.arange(block_size_m)
    offset_on = pid_n * block_size_n + torch.arange(block_size_n)

    # account for strides
    O_ptr = output_ptr + output_.stride(0) * offset_om[:, None] + output_.stride(1) * offset_on[None, :]

    # simulate 1d indexing as in triton
    output_mask = (offset_om[:, None] < M) & (offset_on[None, :] < N)
    tl_store(output_, O_ptr, accumulator, mask=output_mask)

breakpoint() 
expected = torch.matmul(X, Y)
assert torch.allclose(expected, output), breakpoint()
