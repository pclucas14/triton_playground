import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl

N_SKILLS = 2 
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'N_SKILLS': N_SKILLS}, num_stages=1, num_warps=1),    
        ],
    key=['seq_len', 'd_in'],
)
@triton.jit
def poly_linear_kernel(
    # Pointers to matrices
    mixing_coefs_ptr, weight_ptr, input_ptr, output_ptr,
    bs, seq_len, d_in,
    stride_mixing_1, stride_mixing_0, 
    stride_weights_1, stride_weights_0, 
    stride_input_2, stride_input_1, stride_input_0, 
    stride_output_1, stride_output_0, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, N_SKILLS: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    # M == batch_size
    # N == d_out
    # K == d_in
    d_out = 1
    M = seq_len
    N = d_out
    K = d_in

    pid_batch = tl.program_id(axis=0) % bs
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # TODO: check if module is required if masked properly
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + offs_am[:, None] * stride_input_1 + \
                             offs_k[None, :] * stride_input_2 + \
                             pid_batch * stride_input_0

    weight_ptrs = weight_ptr + stride_weights_1 * offs_k[:, None] + offs_bn[None, :] # stride_weights_1 * offs_bn[None, :]

    # additionally, we have to index the mixing_coefs. Here the batch size is the first (index 0) dim.
    mix_ptr = mixing_coefs_ptr + stride_mixing_0 * pid_batch

    # load the block's mixing coefs here, as they do not depend on K
    # mw_mask = (offs_am[:, None] < M)

    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        x_mask   = (offs_am[:, None]  < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        ws_mask  = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)
        x_chunk  = tl.load(input_ptrs, mask=x_mask, other=0.)

        # let's accumulate in for loop, but call only one tl.dot
        weights = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
        for skill_idx in range(N_SKILLS):
            skill_weight_ptr = weight_ptrs + stride_weights_0 * skill_idx
            skill_chunk = tl.load(skill_weight_ptr, mask=ws_mask, other=0.)
            skill_chunk = tl.view(skill_chunk, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            skill_mix_ptr = mix_ptr + stride_mixing_1 * skill_idx
            skill_coef = tl.load(skill_mix_ptr) # , mask=mw_mask, other=0.)
            weights += skill_coef * skill_chunk

        accumulator += tl.dot(x_chunk, weights)
            
        input_ptrs += BLOCK_SIZE_K * stride_input_2
        weight_ptrs += BLOCK_SIZE_K * stride_weights_1

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c = accumulator.to(tl.float16)
    output_ptrs = output_ptr + stride_output_1 * offs_om[:, None] + offs_on[None, :] # + stride_output_1 * offs_on[None, :]
    output_ptrs += stride_output_0 * pid_batch
    output_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(output_ptrs, c, mask=output_mask)

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

    output = torch.empty(bs, seq_len, dtype=DTYPE, device=DEVICE).fill_(torch.nan)
    
    B, M, N = bs, seq_len, 1
    grid = lambda META: (
        B , triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']) ,
    )
    poly_linear_kernel[grid](
        mixing_weights, skill_weights, X, output,
        bs, seq_len, d_in,
        mixing_weights.stride(1), mixing_weights.stride(0), 
        skill_weights.stride(1), skill_weights.stride(0), 
        X.stride(2), X.stride(1), X.stride(0), 
        output.stride(1), output.stride(0)
    )
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * (i * 3) for i in range(2, 5)
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
    bs = 8
    n_skills = N_SKILLS 
    seq_len = 256
    rank = 1
    DTYPE=torch.float16
    DEVICE='cuda'
    # generate weights a la polytropon
    print(f'M : {M}')
    module_logits = torch.randn(bs, n_skills, dtype=DTYPE, device=DEVICE)
    mixing_weights = torch.nn.functional.softmax(module_logits, dim=-1)
    skill_weights = torch.randn(n_skills, M, dtype=DTYPE, device=DEVICE)
    A = torch.einsum("bs,sd->bd", (mixing_weights, skill_weights))

    input = torch.randn(bs, seq_len, M, dtype=DTYPE, device=DEVICE)
    exp_out = torch.einsum('bsi,bi->bs', (input, A))
    triton_out = triton_poly(mixing_weights, skill_weights, input)
    print(f'difference total : {(exp_out - triton_out).abs().sum().item():.8f}')
    print(f'difference max   : {(exp_out - triton_out).abs().max().item():.8f}')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_poly(mixing_weights, skill_weights, input), quantiles=quantiles)
    if provider == 'einsum':
        def _fwd_pass(mixing_weights, skill_weights, input):
            A = torch.einsum("bs,sd->bd", (mixing_weights, skill_weights))
            return torch.einsum('bsi,bi->bs', (input, A))
        ms, min_ms, max_ms = triton.testing.do_bench(lambda : _fwd_pass(mixing_weights, skill_weights, input), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
