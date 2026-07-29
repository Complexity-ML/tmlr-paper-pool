# Token Identity as a Routing Signal for Residual MLP Experts

Manuscripts and reproducibility artifacts for the matched token-identity residual-routing study.

Every token traverses a shared dense SwiGLU branch. A fixed layer-specific token-ID lookup selects two of four narrow residual experts, which process the same contextual hidden state. Token identity controls parameter selection rather than replacing contextual computation.

## Primary matched run

- 306.5M parameters per model
- 8B FineWeb-Edu training tokens
- token-routed evaluation-stream NLL: 2.9329
- dense evaluation-stream NLL: 2.9482
- one seed per architecture
- fixed evaluation stream drawn from the training split
- slower routed training throughput in the reported implementation
- C4 validation NLL: routed 3.4066 versus dense 3.4161
- Pile test-subset NLL: routed 2.8769 versus dense 2.8690

The result is a matched-run observation. C4 and WikiText-2 favour routing while
The Pile subset favours dense, so the measured effect is corpus dependent rather
than evidence of general superiority over dense or learned routing.

## Explore and reproduce

- [Interactive paper companion](https://huggingface.co/spaces/Pacific-i64/Token-Routing-Interactive-Paper)
- [TR-MOE-306 checkpoint](https://huggingface.co/Pacific-i64/TR-MOE-306)
- [Dense-306 checkpoint](https://huggingface.co/Pacific-i64/Dense-306)
- [Linux CPU inference runtime](https://github.com/Complexity-ML/vllm-i64)

The repository includes the TMLR manuscript, Nature-format edition, reviewer response material, standalone implementation, routing tests, raw metrics, downstream results and figure scripts.
