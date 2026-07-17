# Matched 1B-token learned-router/dense comparison

These are the raw logs and run configurations for Panel C. Both runs use seed 42, the same o200k FineWeb-Edu mmap shard, 954 optimizer steps, 1,048,576 tokens per step, and two RTX PRO 6000 Blackwell GPUs.

| Variant | Parameters | Train loss @ 950 | Eval loss @ 750 | Throughput |
|---|---:|---:|---:|---:|
| Learned contextual top-2 + auxiliary balancing | 98,212,820 | 4.790919 | 4.857988 | 95,045 tok/s |
| Dense residual | 98,197,440 | 4.784939 | 4.843310 | 99,993 tok/s |

The language-model train loss excludes the auxiliary term. At step 750, the learned router's auxiliary term is 0.01003241 and its expert traffic is 0.2400/0.2533/0.2610/0.2457, with no dead expert.
