# Exploration: shared dense substrate for conditional modules

## Status

Research hypothesis only. This idea is not part of the current TMLR claims and has not been validated beyond the deterministic lexical-routing instance.

## Motivation

COMPLEXITY-DEEP combines a regular dense path applied to every token with a narrow condition-dependent residual:

$$
\mathbf{y}
=
F_{\mathrm{dense}}(\mathbf{x};\theta_s)
+
\alpha\,G_{\mathrm{conditional}}(\mathbf{x},z;\theta_c).
$$

For the evaluated model:

- $F_{\mathrm{dense}}$ is the shared SwiGLU branch;
- $z$ is token identity;
- $G_{\mathrm{conditional}}$ is the deterministic top-2 lexical residual.

The working hypothesis is that this dense-first decomposition may be useful when integrating conditional or structurally irregular modules into tensor frameworks. The dense path retains a large regular batched computation, while irregular dispatch is confined to a smaller residual branch.

## What “shared” means

Shared means that the same parameterized transformation is applied to all tokens. It does not mean that arbitrary Python objects become shared-memory objects or automatically map to efficient PyTorch kernels.

A new conditional module remains compatible with training only if its state and computation are represented through:

- tensors;
- registered parameters;
- differentiable tensor operations;
- batchable indexing, gather, grouped computation, and scatter;
- specialized kernels when standard operators are insufficient.

## Potential general design pattern

A possible architecture family is:

$$
\text{regular dense substrate}
+
\text{narrow conditional residuals}.
$$

Candidate residual objects include:

- lexical experts;
- retrieval-conditioned adapters;
- memory cells selected by an address;
- graph- or structure-conditioned transformations;
- modality-specific residual modules;
- tool- or state-conditioned adapters;
- recurrent or state-space corrections with discrete state partitions.

The dense substrate may preserve a robust common transformation while the conditional residual introduces structured capacity without forcing the entire layer onto an irregular execution path.

## Non-claims

The current evidence does not show that:

- arbitrary conditional objects benefit from this decomposition;
- the shared path guarantees better optimization or sample efficiency;
- conditional residuals learn semantic specialization;
- the pattern improves wall-clock throughput;
- PyTorch automatically fuses or optimizes the residual objects;
- a dense shared path is necessary for all new module types.

The current routed implementation is slower than dense, so sparse-dispatch overhead remains material.

## Falsifiable experiments

### 1. Shared-width sweep

Hold total stored MLP width constant and vary the allocation between shared and conditional capacity:

- dense-only: $4096+0$;
- shared-heavy: $3840+4\times64$;
- intermediate: $3072+4\times256$;
- routed-heavy: $2048+4\times512$;
- no-shared: $0+4\times1024$.

Measure:

- loss at matched tokens;
- tokens/s;
- GPU utilization;
- kernel launch count;
- dispatch time;
- activation memory;
- stability across seeds.

### 2. Conditional-object comparison

Keep the shared backbone fixed and compare residual conditions:

- token identity;
- random fixed assignment;
- context hash;
- retrieval bucket;
- learned hidden-state routing;
- no conditional residual.

This tests whether the benefit is lexical, conditional in general, or merely additional residual capacity.

### 3. Tensorization ladder

Implement the same conditional module at three levels:

1. Python loops and per-object calls;
2. tensorized gather/group/scatter;
3. grouped GEMM or specialized Triton/CUDA kernel.

Compare outputs for numerical equivalence and measure the execution overhead removed at each level.

### 4. Shared-path necessity

At matched stored parameters and tokens, compare:

- shared plus conditional residual;
- conditional-only;
- dense-only;
- two dense residual branches without routing.

This is required before claiming that the shared path specifically enables useful conditional computation rather than simply providing most of the model capacity.

## Measurements needed for a future paper claim

A defensible general claim would require:

- at least two distinct conditional module families;
- multiple seeds;
- matched-parameter and matched-token controls;
- dense residual controls without routing;
- wall-clock and kernel-level profiling;
- an explicit tensorization description;
- evidence separating architectural quality from implementation overhead.

## Promotion criterion

Promote this idea into the paper only if experiments show that shared-plus-conditional residuals consistently outperform both dense-only and conditional-only controls across more than one conditional object family, without relying on unsupported specialization arguments.
