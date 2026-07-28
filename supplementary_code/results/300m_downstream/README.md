# Final 300M zero-shot benchmark

This directory contains the reviewer-facing aggregate results for the final matched 306.5M-parameter dense and token-identity residual checkpoints trained for 8B tokens.

## Protocol

- checkpoints: step 7,629 for both models;
- checkpoint SHA-256: `0d5bc1d8d8ead8ecde70e2705dfc8712e99a3de79bdf24f9b4e34fd1f668de7b` (dense), `361b73873ac0dda66552325cc31328666b9d4c7684d7041e5b37cd43c42712a2` (token-identity residual);
- runtime: `vllm 0.1.1.dev4+g6f8d5018d.precompiled`, released as Pacific v0.3.1;
- evaluation: EleutherAI LM Evaluation Harness 0.4.12, `transformers` 4.57.6, `datasets` 5.0.0;
- precision/context: BF16, maximum sequence length 2,048;
- decoding backend: CUDA Graph enabled;
- task settings: zero-shot, no sample limit, automatic batching, identical seeds;
- reported multiple-choice metric: `acc_norm`;
- WikiText-2 metric: word perplexity.

The evaluation used the recorded CUDA-graph inference wheel with SHA-256 `81ea56c3d2eaa671f57755efd633d95a219e0960b1964ae70d13fe5a605921b3`. The exact evaluation and export scripts are retained in the blinded reviewer artifact and will be released after review.

`summary.csv` is the compact table source. `aggregated_results.json` retains task sizes, point estimates, standard errors, package versions, and model hashes. The full per-example LM Harness outputs, execution logs, model export metadata, and sanity generations were retained in the local archive `reviewer-bench-artifacts-20260717.tar.gz` (SHA-256 `9fc95fd41f6067e32ec8ff295ac210732941689d087f9689c517b863dc8d386e`). They are not committed because the logged per-example files are approximately 100 MB and are reproducible from the public scripts and checkpoints.
