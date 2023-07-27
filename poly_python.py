import torch
DTYPE = torch.float32
# --- ARGUMENTS

# Pointers to matrices
mixing_weights_ptr, weights_ptr, input_ptr, output_ptr = 0, 0, 0, 0

# Matrix dimensions
bs, n_skills, rank, d_in, d_out = 32, 8, 16, 256, 256

# generate weights a la polytropon
module_logits = torch.randn(bs, n_skills, dtype=DTYPE, device='cuda')
mixing_weights = torch.nn.functional.softmax(module_logits, dim=-1)
skill_weights = torch.randn(n_skills, d_in, rank, dtype=DTYPE, device='cuda')
input_ = torch.randn(bs, d_in, dtype=DTYPE, device='cuda')
output_ = torch.empty(bs, rank, dtype=DTYPE, device='cuda')

# meta parameters
BLOCK_SIZE_M = 16 # 5 
BLOCK_SIZE_N = 16 # 7
BLOCK_SIZE_K = 16 # 3
GROUP_SIZE_M = 16 # 4

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
for pid in range(ceil_div(M, BLOCK_SIZE_M) * ceil_div(N, BLOCK_SIZE_N)):

    # simulate a kernel call
    if use_groups:
        # get better l2-cache utilization by grouping
        num_pid_m = ceil_div(M, BLOCK_SIZE_M)
        num_pid_n = ceil_div(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        GROUP_SIZE_M = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
        print(f'pid_m: {pid_m}, pid_n: {pid_n}')
        pass
    else:
        # each pid computes one (i,j) / (m, n) entry in the final matrix
        num_blocks_n = ceil_div(N, BLOCK_SIZE_N)
        pid_m = pid // num_blocks_n
        pid_n = pid % num_blocks_n

        print(f'pid_m: {pid_m}, pid_n: {pid_n}')

    # compute the offsets into the input matrices
    # this is exactly correct if strides for this dimension are 1. 
    offset_am =  pid_m * BLOCK_SIZE_M + torch.arange(BLOCK_SIZE_M) # this is the input. So ok 
    offset_bn =  pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N) # this is the weights
    offset_k = torch.arange(BLOCK_SIZE_K)
    offset_skills = torch.arange(n_skills)

    # account for strides
    # (BLOCK_SIZE_M, BLOCK_SIZE_K) [These are 2-D indices for X
    # we will progressively move the k index to get the full row
    # this is the input. so ok
    X_ptr = input_ptr + offset_am[:, None] * input_.stride(0) + offset_k[None, :] * input_.stride(1)

    # this is the weight tensor
    Y_ptr = weights_ptr + skill_weights.stride(2) * offset_bn[None, :] + \
                          skill_weights.stride(1) * offset_k[:, None] # + \
                          # skill_weights.stride(0) * offset_skills[:, None, None]

    # additionally, we have to index the mixing_coefs. Here the batch size is the first (index 0) dim.
    mix_ptr = mixing_weights_ptr + mixing_weights.stride(0) * offset_am[:, None] # + mixing_weights.stride(1) * offset_w[None, :]

    accumulator = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=input_.dtype, device=input_.device)
    
    # load the block's mixing coefs here, as they do not depend on K
    mw_mask  = (offset_am[:, None] < M) 
    
    # for k_chunk in range(0, K, BLOCK_SIZE_K):
    for k_chunk in range(0, ceil_div(K, BLOCK_SIZE_K)): # makes the masking of invalid outputs easier

        x_mask   = (offset_am[:, None]  < M) & (offset_k[None, :] < K - k_chunk * BLOCK_SIZE_K)
        ws_mask  = (offset_k[:, None] < K - k_chunk * BLOCK_SIZE_K) & (offset_bn[None, :] < N)

        # simulate 1d indexing as in triton                       # actual shapes          # dimensions being indexed
        x_chunk  = tl_load(input_, X_ptr, mask=x_mask)            # BSM, BSK               # bs,     d_in         (M, K)
        ws_chunk = tl_load(skill_weights, Y_ptr, mask=ws_mask)    # n_skills, BSN, BSK     # skills, d_in, d_out  (_, K, N)
        mw_chunk = tl_load(mixing_weights, mix_ptr, mask=mw_mask) # BSM, n_skills          # bs,     skills       (M, _)

        # let's just iterate over the skills for now
        for skill_idx in range(n_skills):
            skill_weight_ptr = Y_ptr + skill_weights.stride(0) * skill_idx
            skill_chunk = tl_load(skill_weights, skill_weight_ptr, mask=ws_mask).view(BLOCK_SIZE_K, BLOCK_SIZE_N)
            skill_mix_ptr = mix_ptr + mixing_weights.stride(1) * skill_idx 
            skill_weight = tl_load(mixing_weights, skill_mix_ptr, mask=mw_mask)
            # accumulator += mw_chunk[:, skill_idx, None] * torch.matmul(x_chunk, ws_chunk[skill_idx])
            accumulator += skill_weight * torch.matmul(x_chunk, skill_chunk)

        '''
        ws_rshp = ws_chunk.view(n_skills, -1)
        merged_weights = torch.matmul(mw_chunk, ws_rshp)
        merged_weights = merged_weights.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
        x_rshp = x_chunk.view(BLOCK_SIZE_M, 1, BLOCK_SIZE_K)
        out = torch.matmul(x_rshp, merged_weights).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
        accumulator += out
        '''
        
        # move pointers
        X_ptr += BLOCK_SIZE_K * input_.stride(1)
        Y_ptr += BLOCK_SIZE_K * skill_weights.stride(1)

    # expected_tmp = torch.zeros_like(accumulator)
    # x_chunk = input_[offset_am]
    # A_chunk = torch.einsum("bs,sdr->bdr", (mixing_weights[offset_am], skill_weights[:, :, offset_bn]))
    # out = torch.bmm(x_chunk.unsqueeze(1), A_chunk).squeeze()
    # breakpoint()

    # compute the offsets into the output matrix (think of this as 1D-indexing)
    offset_om = pid_m * BLOCK_SIZE_M + torch.arange(BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N)

    # account for strides
    O_ptr = output_ptr + output_.stride(0) * offset_om[:, None] + output_.stride(1) * offset_on[None, :]

    # simulate 1d indexing as in triton
    output_mask = (offset_om[:, None] < M) & (offset_on[None, :] < N)
    tl_store(output_, O_ptr, accumulator, mask=output_mask)

print(f'difference total : {(exp_out - output_).abs().sum().item():.8f}')
print(f'difference max   : {(exp_out - output_).abs().max().item():.8f}')