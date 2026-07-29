# Nature Machine Intelligence editorial readiness

Last audited: 28 July 2026.

## Current package

The manuscript is technically aligned with the journal's initial Article format:

- 127-word unreferenced abstract (limit: 150);
- main text below 3,500 words, excluding Methods, references and legends;
- unheaded introduction followed by Results, Discussion and Methods;
- two figures and four tables (six main display items, the maximum);
- twelve references;
- eight manuscript pages as an internal project constraint, not a journal rule;
- separate Supplementary Information;
- double-anonymized manuscript and code archive;
- generative-AI assistance disclosed in Methods; and
- 19 routing and configuration tests passing locally.

Nature Machine Intelligence accepts compiled PDF for initial submission and does
not accept presubmission enquiries.

## Submission blockers

1. The overlapping TMLR manuscript is still under consideration. Do not submit
   this manuscript to Nature Machine Intelligence until TMLR has issued a final
   decision or the TMLR submission has been formally withdrawn.
2. Complete every author, affiliation, ORCID, contribution, competing-interest
   and contact placeholder in `cover_letter.tex`.
3. Record the final status and identifier of the TMLR manuscript in the cover
   letter.
4. Confirm that editors and reviewers can access the exact checkpoints,
   per-example evaluation records and anonymized source archive.
5. Before publication, deposit the code and supporting records in a persistent,
   DOI-minting repository and update the Data and Code availability statements.

## Editorial risk assessment

The paper is honest and internally coherent, but the present evidence is
high-risk for editorial rejection at a Nature-branded journal:

- the 306.5M comparison has one seed per architecture;
- its fixed loss-evaluation stream comes from the training split;
- the standard-task panel contains only four tasks and shows near parity;
- the two 262,144-token independent-corpus subsets produce opposite model orderings
  and were not decontaminated against FineWeb-Edu;
- the measured effect is modest; and
- the routed implementation is approximately 21% slower at the reported batch.

The strongest broad-interest contribution is therefore not a performance claim.
It is the controlled separation of conditional residual capacity from adaptive
contextual routing.

## Evidence that would materially strengthen the submission

Ordered by expected value:

1. Add multiple seeds for an affordable matched scale and report the distribution
   of routed-minus-dense effects.
2. Expand the preregistered independent-corpus evaluation beyond the current
   C4 and Pile subsets and perform a decontamination audit.
3. Add token-frequency-stratified and memorization analyses, because routing is
   keyed directly to lexical identity.
4. Expand the task panel with stronger out-of-distribution and language-model
   evaluations chosen before observing results.
5. Add a compute- or wall-clock-matched comparison, not only a token-matched one.
6. If resources permit, replicate the primary 8B-token pair.

## Recommended sequence

Continue the TMLR discussion through its current decision window. In parallel,
keep this Nature package current and run the remaining replication and diagnostic
analyses. If TMLR rejects the paper, or if the submission is withdrawn, fill the
author-only fields, rebuild the package, visually inspect every PDF page and then
make the Nature Machine Intelligence submission.

## Official references

- Article format: https://www.nature.com/natmachintell/content
- Initial files and cover-letter requirements:
  https://www.nature.com/natmachintell/submission-guidelines/preparing-your-submission
- Initial PDF formatting:
  https://www.nature.com/natmachintell/submission-guidelines/initial-formatting
- Presubmission enquiries:
  https://www.nature.com/natmachintell/submission-guidelines/presubmission-enquiries
- Code and data policy:
  https://www.nature.com/natmachintell/editorial-policies/reporting-standards
- Overlapping-submission policy:
  https://www.nature.com/nature/editorial-policies
