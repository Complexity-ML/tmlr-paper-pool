# Short-budget learned-router diagnostic

This directory contains the reviewer-facing raw artifacts for the separate 100M-parameter, 99,614,720-token diagnostic reported as Panel B in the paper.

## Protocol

- seed: 42
- tokenizer: `o200k_base` (vocabulary 200,019)
- sequence length: 2,048
- global batch: 512 sequences (`32/GPU x 2 GPUs x accumulation 8`)
- optimizer steps: 95
- evaluation point: step 75
- training point: step 95
- hardware: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition

The learned routers select top-2 experts from contextual hidden states. They do not load or use token-frequency statistics. The fixed lexical control uses exact corpus-derived counts to balance its secondary route. Its throughput is not used for cross-panel claims because the public implementation lacks the historical optimized sparse kernel.

For the auxiliary variant, `train_loss` is language-model loss and `total_loss` additionally includes `router_aux_loss`. The paper reports `train_loss` consistently across variants.

Each run directory contains:

- `run_config.json`: realized, anonymized configuration and parameter count;
- `metrics.csv`: raw per-step training, evaluation, throughput, router, and expert telemetry.

Reconstruct the paper rows with:

```bash
python supplementary_code/scripts/summarize_100m_router_short.py
```

The short panel is a promotion diagnostic. It uses the same o200k tokenizer family as the historical 1B-token/B200 panel, but must not be compared numerically with it because the data ingestion and evaluation streams, token budget, hardware, and implementation snapshot differ.
