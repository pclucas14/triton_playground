import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['bs', 'd_in', 'd_out'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    x_ptr, W_ptr, out_ptr, lora_a_ptr, lora_b_ptr, 
    # Matrix dimensions
    bs, d_in, d_out,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_x_bs, stride_x_din,
    stride_W_din, stride_W_dout,
    stride_out_bs, stride_out_dout,
    stride_lora_a, stride_lora_b,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(bs, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(d_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_bs = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % bs
    offs_dout = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % d_out
    offs_din = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_bs[:, None] * stride_x_bs + offs_din[None, :] * stride_x_din)
    W_ptrs = W_ptr + (offs_din[:, None] * stride_W_din + offs_dout[None, :] * stride_W_dout)

    lora_a_ptrs = lora_a_ptr + offs_din[None, :] * stride_lora_a
    lora_b_ptrs = lora_b_ptr + offs_dout[None, :] * stride_lora_b

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    lora_b = tl.load(lora_b_ptrs, mask=offs_dout[None, :] < d_out, other=0.0)
    running_lora_dot = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(d_in, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        x_chunk = tl.load(x_ptrs, mask=offs_din[None, :] < d_in - k * BLOCK_SIZE_K, other=0.0)
        W_chunk = tl.load(W_ptrs, mask=offs_din[:, None] < d_in - k * BLOCK_SIZE_K, other=0.0)
        
        # (1, K) so we can broadcast on M
        lora_a = tl.load(lora_a_ptrs, mask=offs_din[None, :] < d_in - k * BLOCK_SIZE_K, other=0.0)
        running_lora_dot += tl.sum(lora_a * x_chunk, axis=1)

        #W_chunk = W_chunk + lora_a[:, None] #* lora_b[None, :] # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        # standard matmul 
        accumulator += tl.dot(x_chunk, W_chunk)

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_x_din
        W_ptrs += BLOCK_SIZE_K * stride_W_din

        lora_a_ptrs += BLOCK_SIZE_K * stride_lora_a

    # apply rank-1 lora 
    accumulator += running_lora_dot[:, None] * lora_b 
    
    accumulator = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_out_bs * offs_cm[:, None] + stride_out_dout * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < bs) & (offs_cn[None, :] < d_out)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def matmul(x, W, lora_a, lora_b):
    # Check constraints.
    assert x.shape[1] == W.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix A must be contiguous"
    assert W.is_contiguous(), "Matrix B must be contiguous"
    assert lora_a.ndim == lora_b.ndim == 1
    bs, d_in = x.shape
    d_in, d_out = W.shape

    assert lora_a.size(0) == d_in
    assert lora_b.size(0) == d_out

    # Allocates output.
    output = torch.empty((bs, d_out), device=x.device, dtype=x.dtype).fill_(0)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(bs, META['BLOCK_SIZE_M']) * triton.cdiv(d_out, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        x, W, output, lora_a, lora_b, 
        bs, d_in, d_out,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        output.stride(0), output.stride(1),
        lora_a.stride(0), lora_b.stride(0),
    )
    return output


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d_in', 'd_out'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            512, 1024, 2048, 4096, 4096 * 2
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(d_in, d_out, provider):
    bs = 8
    d_in = d_out // 2
    print(bs, d_in, d_out)
    x = torch.randn((bs, d_in), device='cuda', dtype=torch.float16)
    W = torch.randn((d_in, d_out), device='cuda', dtype=torch.float16)
    
    lora_a = torch.randn((d_in,), device='cuda', dtype=torch.float16)
    lora_b = torch.randn((d_out,), device='cuda', dtype=torch.float16)#.fill_(0)

    adapter_fwd = lambda: F.linear(x, W.T) + x.matmul(lora_a).view(-1, 1) * lora_b.view(1, -1)
    exp_out = F.linear(x,W.T) + x.matmul(lora_a[:, None]).matmul(lora_b[None, :])
    triton_out = matmul(x, W, lora_a, lora_b)

    diff_total = (exp_out - triton_out).abs().sum()
    diff_max = (exp_out - triton_out).abs().max()
    diff_mean = (exp_out - triton_out).abs().mean()
    print(f'difference total : {diff_total.item():.8f}')
    print(f'difference max   : {diff_max.item():.8f}')
    print(f'difference mean  : {diff_mean.item():.8f}')
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(adapter_fwd, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(x, W, lora_a, lora_b), quantiles=quantiles)
    perf = lambda ms: 2 * bs * d_out * d_in * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
