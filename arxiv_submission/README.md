# Simplified arXiv edition

This directory builds a compact two-column preprint that is separate from the
double-blind TMLR and Nature-format manuscripts.

The preprint keeps:

- the shared-plus-routed architecture;
- the matched 306.5M / 8B-token comparison;
- C4 and Pile independent-corpus NLL;
- the recent parameter-matched Apple-MPS GQA/MHA pilots;
- explicit limitations and reproducibility links.

It omits the long reviewer-oriented discussion and historical ablation panels
whose protocols are not directly comparable with the recent local pilots.

Run:

```bash
bash build_arxiv.sh
```

Outputs:

- `output/pdf/token_identity_routing_arxiv.pdf`
- `dist/token_identity_routing_arxiv_source.zip`

The upload archive contains only the TeX source, bibliography, and figure
required to compile the paper.
