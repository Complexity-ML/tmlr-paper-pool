# OpenReview revision fields

## Title

Token Identity as a Routing Signal for Residual MLP Experts

## Abstract

Learned mixture-of-experts routers use contextual hidden states to decide which parameters process each token. We ask how much of that routing signal can come from token identity alone when expert capacity is a small residual over a shared dense MLP. Our model maps each token ID to two narrow experts through a fixed, layer-specific lookup; the experts still transform the same contextual hidden state as the shared branch. In a matched single-seed comparison between 306.5M-parameter models trained on 8B FineWeb-Edu tokens, the token-routed model reaches diagnostic-stream NLL 2.9329 versus 2.9482 for dense, although the stream comes from the training split and routing is slower. Final checkpoints remain close on three zero-shot tasks. The routed model has lower WikiText-2 perplexity and C4-validation NLL, whereas dense has lower NLL on a Pile test subset. A learned-router diagnostic also changes ordering between short and 1.0003B-token budgets. The observed effect is therefore modest and corpus dependent; replication across seeds remains open.

## Submission Type

Regular submission (12 pages or fewer of main content)

Rationale: the revised paper has eight pages of main content; references begin on PDF page 9.

## Beyond PDF

Leave empty. The supplementary ZIP does not belong in this field.

## Previous TMLR Submission URL

https://openreview.net/forum?id=jZq6EVboC6

## Changes Since Last Submission

This revision aligns the paper with the architecture and measurements supported by the reported runs and substantially improves the presentation.

The main changes are:

1. **Focused the research question.** The paper now asks whether token identity can usefully select a small residual parameter subspace when a shared dense path already performs common contextual computation. It does not present fixed lookup as a replacement for contextual processing or learned MoE routing.

2. **Aligned the architecture and claims.** Mu-Guidance and unsupported orthogonalization, universal-approximation, independent-expert-objective, and semantic-specialization claims are no longer part of the active paper. The evaluated architecture is a shared dense SwiGLU branch plus a small fixed top-2 lexical residual branch.

3. **Specified the realized routing mechanism.** In the primary model, each layer uses a seeded permutation of token ID modulo four for the primary expert and its cyclic successor as the secondary expert. Two distinct experts are combined with fixed 0.5/0.5 weights; shared/routed branch gates were initialized to 0.5/0.5. The primary model has no learned router or auxiliary load-balancing loss; learned contextual routers are evaluated separately as controls.

4. **Simplified the analysis.** The revised method retains exact routing and capacity accounting but removes the elementary proposition and the uninformative balance figure. The architecture diagram now isolates the central distinction: token identity controls selection, while both branches transform the same contextual hidden state.

5. **Reorganized the empirical evidence.** The matched 306.5M-parameter, 8B-token comparison is the primary result. At the last common evaluation checkpoint, the token-routed model reaches evaluation-stream NLL 2.9329 versus 2.9482 for dense. Training throughput on the reported 8xB300 system is presented separately because dense remains faster in the current implementation.

6. **Expanded the 100M controls.** The paper retains all seven historical approximately 100M-parameter, 1.0003B-token ablations and identifies their realized lookup from preserved configurations and launcher source. Their checkpoints were not retained, so new corpus evaluation requires rerunning them. The paper also adds a four-condition short-budget learned-router diagnostic and a matched 1.0003B-token learned-router/dense comparison.

7. **Corrected and expanded evaluation.** The fixed evaluation stream is identified as coming from the FineWeb-Edu training split rather than an independent holdout. The paper explains the different final train/evaluation steps, no longer uses the final-50-step average as headline evidence, and adds zero-shot ARC-Easy, PIQA, HellaSwag, and WikiText-2 evaluations. It also adds paired 262,144-token measurements on pinned C4-validation and Pile-test subsets; routing leads on C4 while dense leads on The Pile.

8. **Made the limitations explicit.** The evidence remains single-seed, the primary evaluation stream is not held out, the downstream task differences are small, and the routed implementation is slower than dense. The learned-router controls use a separate data shard and ingestion pipeline, implementation snapshot, and hardware, and multi-seed replication remains open.

9. **Rebuilt the reproducibility package.** The anonymized supplementary ZIP contains the standalone PyTorch implementation, routing tests, run configurations, raw metrics, figure-generation scripts, the verified 32k tokenizer, the learned-router configurations, and downstream aggregate results. It has no dependency on a private framework.

## Competing Interests

N/A

## Human Subjects Reporting

N/A

## Files to upload

PDF:
complexity_deep.pdf

Supplementary Material:
dist/complexity_deep_tmlr_supplement.zip

Do not add a repository URL to the anonymous OpenReview submission.
