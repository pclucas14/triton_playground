import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl

DEBUG = False
N_SKILLS = 2
if DEBUG: 
    config = [
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 128, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
    ]        
else:
    config=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 128, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 256, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 512, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 1024, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=8),    
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 2048, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 4096, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 512, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 1024, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=8),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 2048, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 4096, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=4),    
    ]
@triton.autotune(
    key=['bs', 'seq_len', 'd_in'],
    configs=config,
)
@triton.jit
def poly_linear_kernel(
    # Pointers to matrices
    alpha_ptr, weight_ptr, input_ptr, output_ptr,
    bs, seq_len, d_in,
    stride_alpha_b, stride_alpha_k,      # bs, n_skills 
    stride_weights_k, stride_weights_d,  # n_skills, d_in
    stride_input_bs, stride_input_d,     # (bs), seq_len, d_in 
    stride_output_b, stride_output_s,    # bs, seq_len 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, N_SKILLS: tl.constexpr
):
    K = d_in
    M = bs * seq_len
    
    pid_per_batch = tl.cdiv(seq_len, BLOCK_SIZE_M)
    batch_idx = tl.program_id(axis=0) // pid_per_batch
    seq_idx = tl.program_id(axis=0) % pid_per_batch

    # ----------------------------------------------------------
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(bs * seq_len, d_in), # TODO : move from bs * seq_len to just seq_len,
        strides=(stride_input_bs, stride_input_d), # .. and adjust strides accordingly
        offsets=(batch_idx * seq_len + seq_idx * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1,0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(N_SKILLS, d_in),
        offsets=(0, 0),
        strides=(stride_weights_k, stride_weights_d),
        block_shape=(1, BLOCK_SIZE_K),
        order=(1,0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(bs, seq_len),
        strides=(stride_output_b, stride_output_s), # stride_output_s),
        offsets=(batch_idx, seq_idx * BLOCK_SIZE_M), #(batch_idx, seq_idx), TODO: WHY * BLOCK_SIZE_M ?
        block_shape=(1, BLOCK_SIZE_M), #(1, BLOCK_SIZE_M),
        order=(0,1),
    )
    # first index is 

    ''' 
    offs_am = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs  = input_ptr + offs_am[:, None] * stride_input_0 + offs_k[None, :] * stride_input_1
    weight_ptrs = weight_ptr + offs_k[None, :] * stride_weights_1 

    # load the block's mixing coefs here, as they do not depend on K
    # mw_mask = (offs_am[:, None] < M)
    '''
    # additionally, we have to index the mixing_coefs. Here the batch size is the first (index 0) dim.
    mix_ptr = alpha_ptr + batch_idx * stride_alpha_b
    
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        '''
        x_mask   = (offs_am[:, None]  < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        ws_mask  = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        x_chunk  = tl.load(input_ptrs, mask=x_mask, other=0.)
        '''
        x_chunk = tl.load(input_block_ptr, boundary_check=(0,1))
        
        # let's accumulate in for loop, but call only one tl.dot
        weights = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float16)
        for skill_idx in range(N_SKILLS):
            skill_mix_ptr = mix_ptr + stride_alpha_k * skill_idx
            skill_coef = tl.load(skill_mix_ptr)
            skill_chunk = tl.load(weight_block_ptr, boundary_check=(0,))

            '''
            skill_weight_ptr = weight_ptrs + stride_weights_0 * skill_idx
            skill_chunk = tl.load(skill_weight_ptr, mask=ws_mask, other=0.)
            skill_mix_ptr = mix_ptr + stride_alpha_1 * skill_idx
            '''
            weights += skill_coef * skill_chunk
            weight_block_ptr = tl.advance(weight_block_ptr, (1,0))

        accumulator += tl.sum(x_chunk * weights, axis=1) #tl.dot(x_chunk, weights)

        ''' 
        weight_ptrs += BLOCK_SIZE_K * stride_weights_1
        input_ptrs  += BLOCK_SIZE_K * stride_input_1
        '''
        # should be no need now to increment the weight_block_ptr
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_SIZE_K))
        weight_block_ptr = tl.advance(weight_block_ptr, (-N_SKILLS, BLOCK_SIZE_K))

    '''
    offs_om = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c = accumulator.to(tl.float16)
    output_ptrs = output_ptr + stride_output_0 * offs_om# [:, None]
    output_mask = (offs_om < M)
    tl.store(output_ptrs, c, mask=output_mask)
    '''
    tl.store(output_block_ptr, accumulator.to(tl.float16)[None, :], boundary_check=(0,1))
    # tl.store(output_block_ptr, accumulator.to(tl.float16), boundary_check=(0,))
    
def triton_poly(mixing_weights, skill_weights, X):

    # What has rank ? 
    # output : bs, seq_len, rank
    # weight : n_skills, d_in, rank

    bs, n_skills = mixing_weights.size()
    assert skill_weights.size(0) == N_SKILLS

    n_skills, d_in = skill_weights.size()
    assert X.size(0) == bs and X.size(2) == d_in
    seq_len = X.size(1)

    DTYPE = mixing_weights.dtype
    DEVICE = mixing_weights.device

    output = torch.empty(bs, seq_len, dtype=DTYPE, device=DEVICE)
    # output = torch.empty(bs * seq_len, dtype=DTYPE, device=DEVICE)# .fill_(torch.nan)
    X = X.flatten(0,1)

    B, M, N = bs, seq_len, 1
    grid = lambda META: (
        bs * triton.cdiv((seq_len), META['BLOCK_SIZE_M']),
    )
    poly_linear_kernel[grid](
        mixing_weights, skill_weights, X, output,
        bs, seq_len, d_in,
        mixing_weights.stride(0), mixing_weights.stride(1), 
        skill_weights.stride(0), skill_weights.stride(1), 
        X.stride(0), X.stride(1), 
        output.stride(0), output.stride(1)#, output.stride(0)
    )
    return output.view(bs, seq_len)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            1024, 2048, 4096, 8192, 16384
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['triton', 'einsum'],
        # Label name for the lines
        line_names=['triton', 'einsum'],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    print(provider)
    bs = 64
    n_skills = N_SKILLS 
    seq_len = 4_096
    # seq_len = 1_024
    rank = 1
    DTYPE=torch.float16
    DEVICE='cuda'
    # generate weights a la polytropon
    print(f'M : {M}')
    module_logits = torch.randn(bs, n_skills, dtype=DTYPE, device=DEVICE)
    mixing_weights = torch.nn.functional.softmax(module_logits, dim=-1)
    skill_weights = torch.randn(n_skills, M, dtype=DTYPE, device=DEVICE)
    input = torch.randn(bs, seq_len, M, dtype=DTYPE, device=DEVICE)
     
    A = torch.einsum("bs,sd->bd", (mixing_weights, skill_weights))

    exp_out = torch.einsum('bsi,bi->bs', (input, A))
    gold_out = torch.einsum('bsi,bi->bs', (input.to(torch.float64), A.to(torch.float64))).to(DTYPE)
    triton_out = triton_poly(mixing_weights, skill_weights, input)
    print(f'diff exp total    : {(exp_out - gold_out).abs().sum().item():.5f}')
    print(f'diff exp max      : {(gold_out - exp_out).abs().max().item():.5f}')
    print(f'diff triton total : {(gold_out - triton_out).abs().sum().item():.5f}')
    print(f'diff triton max   : {(gold_out - triton_out).abs().max().item():.5f}')
    show = lambda idx : ((triton_out[idx] - exp_out[idx]).pow(2) * 100).int()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_poly(mixing_weights, skill_weights, input), quantiles=quantiles)
        #ms, min_ms, max_ms = 1, 1, 1
    if provider == 'einsum':
        def _fwd_pass(mixing_weights, skill_weights, input):
            A = torch.einsum("bs,sd->bd", (mixing_weights, skill_weights))
            return torch.einsum('bsi,bi->bs', (input, A))
        ms, min_ms, max_ms = triton.testing.do_bench(lambda : _fwd_pass(mixing_weights, skill_weights, input), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True)
