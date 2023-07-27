import torch
from utils import tl_cdiv, tl_load, tl_store

# --- ARGUMENTS
# Pointers to matrices
DTYPE = torch.float32
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
BLOCK_SIZE_M = 5 
BLOCK_SIZE_N = 7
BLOCK_SIZE_K = 3
GROUP_SIZE_M = 4

A = torch.einsum("bs,sdr->bdr", (mixing_weights, skill_weights))
A = A.reshape(bs, d_in, rank)
exp_out = torch.einsum('bi,bio->bo', (input_, A))

use_groups = True

M, N = output_.size()
K = d_in

seen = {}
# simulate launch grid
for pid in range(tl_cdiv(M, BLOCK_SIZE_M) * tl_cdiv(N, BLOCK_SIZE_N)):

    # simulate a kernel call
    if use_groups:
        # get better l2-cache utilization by grouping
        num_pid_m = tl_cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl_cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        print(f'pid_m: {pid_m}, pid_n: {pid_n}, pid: {pid}')
        assert (pid_m, pid_n) not in seen, breakpoint()
        seen[(pid_m, pid_n)] = True
        pass
    else:
        # each pid computes one (i,j) / (m, n) entry in the final matrix
        num_blocks_n = tl_cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_blocks_n
        pid_n = pid % num_blocks_n

        print(f'pid_m: {pid_m}, pid_n: {pid_n}, pid: {pid}')

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
    for k_chunk in range(0, tl_cdiv(K, BLOCK_SIZE_K)): # makes the masking of invalid outputs easier

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

        # move pointers
        X_ptr += BLOCK_SIZE_K * input_.stride(1)
        Y_ptr += BLOCK_SIZE_K * skill_weights.stride(1)

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