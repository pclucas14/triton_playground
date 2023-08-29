import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl
from simplified_poly_A_r1 import triton_poly as poly_lora_A

N_SKILLS = 2
DEBUG = False
VERBOSE = False
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
    alpha_ptr, A_ptr, B_ptr, input_ptr, output_ptr,
    bs, seq_len, d_in, d_out,
    stride_alpha_b, stride_alpha_k,      # bs, n_skills 
    stride_A_k, stride_A_d,              # n_skills, d_in
    stride_B_k, stride_B_d,              # n_skills, d_out
    stride_input_b, stride_input_s, stride_input_d,     # (bs), seq_len, d_in 
    stride_output_b, stride_output_s, stride_output_d,  # bs, seq_len, d_out 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, N_SKILLS: tl.constexpr
):
    K = d_in
    M = bs * seq_len
    
    batch_idx = tl.program_id(axis=0) 
    seq_idx = tl.program_id(axis=1)
    
    # batch_size scaling happens here
    input_offset = batch_idx * stride_input_b
    output_offset = batch_idx * stride_output_b

    # ----------------------------------------------------------
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + input_offset,
        shape=(seq_len, d_in), # TODO : move from bs * seq_len to just seq_len,
        strides=(stride_input_s, stride_input_d), # .. and adjust strides accordingly
        offsets=(seq_idx * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1,0),
    )
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(N_SKILLS, d_in),
        offsets=(0, 0),
        strides=(stride_A_k, stride_A_d),
        block_shape=(1, BLOCK_SIZE_K),
        order=(1,0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(N_SKILLS, d_in),
        offsets=(0, 0),
        strides=(stride_B_k, stride_B_d),
        block_shape=(1, BLOCK_SIZE_K),
        order=(1,0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + output_offset,
        shape=(seq_len, d_out),
        strides=(stride_output_s, stride_output_d),
        offsets=(seq_idx * BLOCK_SIZE_M, 0), 
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), 
        order=(0,1),
    )
    # first index is 
    # additionally, we have to index the mixing_coefs. Here the batch size is the first (index 0) dim.
    mix_ptr = alpha_ptr + batch_idx * stride_alpha_b
    
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        x_chunk = tl.load(input_block_ptr, boundary_check=(0,1))
        
        # let's accumulate in for loop, but call only one tl.dot
        weights = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float16)
        for skill_idx in range(N_SKILLS):
            skill_mix_ptr = mix_ptr + stride_alpha_k * skill_idx
            skill_coef = tl.load(skill_mix_ptr)
            skill_chunk = tl.load(A_block_ptr, boundary_check=(0,))

            weights += skill_coef * skill_chunk
            A_block_ptr = tl.advance(A_block_ptr, (1,0))

        accumulator += tl.sum(x_chunk * weights, axis=1) #tl.dot(x_chunk, weights)

        # should be no need now to increment the A_block_ptr
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_SIZE_K))
        A_block_ptr = tl.advance(A_block_ptr, (-N_SKILLS, BLOCK_SIZE_K))

    # -----------------------------------------------------------
    # at this point, we have  a (BLOCK_SIZE_M,) vector in accumulator
    # we iterate over d_out in chunks of (BLOCK_SIZE_K,), and 
    # outer prod to get (BLOCK_SIZE_M, BLOCK_SIZE_K) matrix to be stored
    for k in range(0, tl.cdiv(d_out, BLOCK_SIZE_K)):
        weights = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float16)
        for skill_idx in range(N_SKILLS):
            skill_mix_ptr = mix_ptr + stride_alpha_k * skill_idx
            skill_coef = tl.load(skill_mix_ptr)
            skill_chunk = tl.load(B_block_ptr, boundary_check=(0,))

            weights += skill_coef * skill_chunk
            B_block_ptr = tl.advance(B_block_ptr, (1,0)) 

        # outer prod
        outer_prod = accumulator[:, None] * weights
        tl.store(output_block_ptr, outer_prod.to(tl.float16), boundary_check=(0,1))

        # increment pointers
        output_block_ptr = tl.advance(output_block_ptr, (0, BLOCK_SIZE_K))
        B_block_ptr = tl.advance(B_block_ptr, (-N_SKILLS, BLOCK_SIZE_K))

    # tl.store(output_block_ptr, accumulator.to(tl.float16)[None, :], boundary_check=(0,1))
    # tl.store(output_block_ptr, accumulator.to(tl.float16), boundary_check=(0,))
    
def triton_poly(mixing_weights, A_weights, B_weights, X):

    bs, n_skills = mixing_weights.size()
    assert A_weights.size(0) == N_SKILLS

    n_skills, d_in = A_weights.size()
    n_skills, d_out = B_weights.size()
    assert X.size(0) == bs and X.size(2) == d_in
    seq_len = X.size(1)

    DTYPE = mixing_weights.dtype
    DEVICE = mixing_weights.device

    output = torch.empty(bs, seq_len, d_out, dtype=DTYPE, device=DEVICE).fill_(-1)

    grid = lambda META: (
        bs, triton.cdiv((seq_len), META['BLOCK_SIZE_M'])
    )
    poly_linear_kernel[grid](
        mixing_weights, A_weights, B_weights, X, output,
        bs, seq_len, d_in, d_out,
        mixing_weights.stride(0), mixing_weights.stride(1), 
        A_weights.stride(0), A_weights.stride(1), 
        B_weights.stride(0), B_weights.stride(1), 
        X.stride(0), X.stride(1), X.stride(2), 
        output.stride(0), output.stride(1), output.stride(2),
    )
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  
        x_vals=[
            1024, 2048, 4096, 8192, 16384
        ],  
        line_arg='provider',  
        # Possible values for `line_arg`
        line_vals=['triton', 'einsum', 'mix'],
        # Label name for the lines
        line_names=['triton', 'einsum', 'mix'],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  
        args={},
    )
)
def benchmark(M, N, K, provider):
    print(provider)
    bs = 4
    n_skills = N_SKILLS 
    seq_len = 4_096
    seq_len = 1_024
    rank = 1
    DTYPE=torch.float16
    DEVICE='cuda'
    # generate weights a la polytropon
    print(f'M : {M}')
    module_logits = torch.randn(bs, n_skills, dtype=DTYPE, device=DEVICE)
    mixing_weights = torch.nn.functional.softmax(module_logits, dim=-1)
    A_weights = torch.randn(n_skills, M, dtype=DTYPE, device=DEVICE)
    B_weights = torch.randn(n_skills, M, dtype=DTYPE, device=DEVICE)
    input = torch.randn(bs, seq_len, M, dtype=DTYPE, device=DEVICE)
     
    def _fwd_pass(alpha, A_weights, B_weights, input, dtype=None):
        dtype_out = alpha.dtype
        if dtype is not None:
            alpha, A_weights, B_weights, input = [
                x.to(dtype) for x in [alpha, A_weights, B_weights, input]
            ]
        A = torch.einsum("bs,si->bi", (alpha, A_weights))
        B = torch.einsum("bs,so->bo", (alpha, B_weights))
        tmp = torch.einsum('bsi,bi->bs', (input, A))
        out = torch.einsum('bs,bo->bso', (tmp, B))
        if dtype is not None:
            out = out.to(dtype_out)
        return out 

    def mix_fwd(mixing_weights, A_weights, B_weights, input):
        A_out = poly_lora_A(mixing_weights, A_weights, input)
        B = torch.einsum("bs,so->bo", (mixing_weights, B_weights))
        out = torch.einsum('bs,bo->bso', (A_out, B)) 
        return out


    if VERBOSE:
        exp_out  = _fwd_pass(mixing_weights, A_weights, B_weights, input)
        gold_out = _fwd_pass(mixing_weights, A_weights, B_weights, input, dtype=torch.float64)
        triton_out = triton_poly(mixing_weights, A_weights, B_weights, input)
        print(f'diff exp total    : {(exp_out - gold_out).abs().sum().item():.5f}')
        print(f'diff exp max      : {(gold_out - exp_out).abs().max().item():.5f}')
        print(f'diff triton total : {(gold_out - triton_out).abs().sum().item():.5f}')
        print(f'diff triton max   : {(gold_out - triton_out).abs().max().item():.5f}')
    show = lambda idx : ((triton_out[idx] - exp_out[idx]).pow(2) * 100).int()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_poly(mixing_weights, A_weights, B_weights, input), quantiles=quantiles)
    if provider == 'einsum':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda : _fwd_pass(mixing_weights, A_weights, B_weights, input), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    if provider == 'mix':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda : mix_fwd(mixing_weights, A_weights, B_weights, input), quantiles=quantiles)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
