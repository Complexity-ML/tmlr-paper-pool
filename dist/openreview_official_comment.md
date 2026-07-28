# Draft Official Comment — Response to Reviewer Paiy

We thank the reviewer for the careful reading and concrete criticism. We agree that the previous version did not make the evaluation protocol, empirical scope, and relationship to learned routing sufficiently clear. We have revised the manuscript and supplementary material accordingly. Below we respond point by point.

**1. Figure quality and the undefined “COMPLEXITY-DEEP” label.**

We agree. The “COMPLEXITY-DEEP” label has been removed. The architecture figure has been replaced with a new diagram that directly represents the implemented computation: token identity determines only the fixed top-2 expert selection, while the shared SwiGLU branch and the selected residual experts transform the same contextual hidden state. The new figure uses the notation defined in the Architecture section and is provided as a vector graphic. We also rebuilt the loss figure so that training NLL and the fixed evaluation-stream NLL gap are shown in separate, explicitly labeled panels.

**2. Inconsistent train/evaluation terminology.**

We agree that the former terminology was ambiguous. The revised manuscript now distinguishes the two quantities throughout:

- “Train” denotes language-model NLL recorded on the training stream.
- “Eval. stream” denotes NLL on a fixed stream drawn from the FineWeb-Edu training split.

The abstract, Experimental Design section, Figure 2, the primary Results table, and the 100M ablation captions now use this terminology consistently. The manuscript also states explicitly that this fixed stream is not an independent held-out set.

**3. Step 950 training loss versus step 750 evaluation loss.**

We have added the missing explanation to the 100M Ablations section. The 1B-token runs have a 954-step training budget, record training metrics every 10 steps, and evaluate every 750 steps. Consequently, step 950 is the last common logged training point and step 750 is the last common evaluation checkpoint; the table does not substitute one metric or checkpoint for another. The short diagnostic runs analogously end at step 95, log training loss every five steps, and evaluate every 75 steps.

**4. Missing learned-router comparison.**

We agree that this comparison was necessary to evaluate the motivation. The revised paper adds two learned-router controls:

- A four-condition, 99.6M-token diagnostic comparing dense residual computation, fixed token-identity routing, learned contextual top-2 routing with auxiliary balancing, and learned contextual top-2 routing with loss-free balancing.
- A matched 1.0003B-token comparison between the promoted auxiliary-balanced learned router and its dense control.

At the short budget, the auxiliary-balanced learned router has 0.0119 lower evaluation NLL than dense. This advantage does not persist at 1.0003B tokens: the dense control has 0.0147 lower evaluation NLL and approximately 5.2% higher throughput. We report both outcomes rather than using only the favorable short-budget result. Because these controls use a separate data shard and ingestion pipeline, implementation snapshot, and hardware, they are presented as a separate diagnostic rather than as a direct ranking against the historical 300M run.

**5. Single-seed evidence.**

We agree that the primary comparison cannot establish a robust average effect without replication. We have not added a three-seed replication of the 8B-token, 306.5M-parameter experiment, and we now state this limitation explicitly in the abstract, Introduction, Results, Discussion and Limitations, and Conclusion. We have also narrowed the main claim to a matched-run observation: in this run pair, fixed token-identity routing selected a useful residual parameter subspace. We do not claim statistical significance or general superiority over dense or learned routing.

**6. Evaluation on the training split and generalization.**

We agree. The fixed FineWeb-Edu stream is now identified everywhere as coming from the training split and as not being a held-out evaluation. To provide evidence beyond that stream, we evaluated both final 300M checkpoints with the same zero-shot protocol on ARC-Easy, PIQA, HellaSwag, and WikiText-2. These evaluations do not replace broader out-of-distribution or multi-seed testing, and the revised limitations state this directly.

**7. Held-out perplexity and downstream evaluation.**

The revision adds complete, untruncated zero-shot evaluations of the two final checkpoints. The token-routed model is 0.46 percentage points below dense on ARC-Easy, tied at the displayed precision on PIQA, 0.39 points above dense on HellaSwag, and has 0.59 lower WikiText-2 word perplexity. Each multiple-choice accuracy difference is smaller than one reported standard error. We therefore interpret this panel as near-parity, not as evidence of downstream superiority.

**Additional reproducibility changes.**

The anonymous supplementary archive has been rebuilt to include the standalone PyTorch implementation, routing tests, realized run configurations, raw metrics, learned-router configurations and results, downstream aggregate results, figure-generation scripts, and the verified tokenizer artifacts. The unsupported Mu-Guidance, orthogonality, universal-approximation, independent-expert-objective, and semantic-specialization claims have also been removed from the active paper.

We appreciate the review: it led us to narrow the claim, correct the evaluation language, add the missing learned-router and downstream controls, and make the remaining limitations substantially more explicit.
