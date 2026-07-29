# OpenReview update after the submitted revision

## Title

Correction to the routing description and independent-corpus evaluation

## Comment

After uploading the revision, we performed an additional checkpoint-level audit and identified two descriptive errors in the manuscript. We report them transparently here and correct them in the replacement PDF:

1. In the primary 306.5M model, the stored lookup gives the layer-specific permuted-modulo primary expert. The second expert is its cyclic successor,
   \(r_{l,2}(t)=(r_{l,1}(t)+1)\bmod 4\), rather than a separately stored least-used/cardinality-balanced assignment.
2. The learned shared and routed branch gates were initialized to 0.5/0.5, not 1.0/0.1.

These corrections are verified against the historical training code, checkpoint configuration, stored primary lookup buffers and exported evaluation implementation. They do not change the model weights, tensor shapes, fixed 0.5/0.5 expert-combination weights or previously reported loss and benchmark measurements.

We also added paired independent-corpus evaluation of the final checkpoints, using the exact 32k tokenizer and Apple MLX/Metal. Each corpus contains 262,144 scored targets in 128 fixed blocks:

- C4 validation: token-routed NLL 3.4066 versus dense 3.4161; routed-minus-dense \(-0.0095\), paired-bootstrap 95% CI \([-0.0127,-0.0063]\).
- The Pile test subset: token-routed NLL 2.8769 versus dense 2.8690; routed-minus-dense \(+0.0079\), 95% CI \([+0.0022,+0.0138]\).

No document-level decontamination against FineWeb-Edu was performed. Together with the modest WikiText-2 advantage for routing, the opposite C4 and Pile orderings lead us to narrow the conclusion further: fixed token identity provides a measurable but corpus-dependent residual-routing signal in this single-seed run pair, not a general advantage over dense computation.

