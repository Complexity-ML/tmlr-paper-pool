# Nature Machine Intelligence submission checklist

## Verified locally

- Content type: Article.
- Double-anonymized reviewer manuscript and Supplementary Information.
- Abstract: below the 150-word limit.
- Main text: below the 3,500-word limit, excluding Methods, references and legends.
- Main display items: 2 figures and 4 tables (6 total).
- Structure: unheaded introduction, Results, Discussion and Methods.
- References: 10.
- AI-tool use disclosed in Methods; no AI-generated publication image is used.
- Architecture figure is vector PDF generated from the checked-in Graphviz source.
- Primary final checkpoints: step 7,629 (7.999586304 billion tokens).
- Final training-NLL row: step 7,620 because metrics were logged every ten steps.
- Short contextual-router diagnostic: 99.6 million tokens, training NLL at step 95 and evaluation NLL at step 75.
- Full contextual-router control: 1.0003 billion tokens, training NLL at step 950 and evaluation NLL at step 750.
- CPU routing and control tests: 19 passed.

## Required before submission

- Fill every author and contact field in `cover_letter.tex`.
- Confirm the author-contribution, funding and competing-interest declarations.
- Withdraw the overlapping TMLR submission and retain written confirmation.
- Replace the related-manuscript placeholder with the TMLR withdrawal date and identifier.
- Confirm whether any prior discussion occurred with a Nature Machine Intelligence editor.
- Run `./build_submission.sh` and visually inspect all final PDFs.
- In the portal, select double-anonymized peer review and enter the minimal competing-interest statement.
- Upload the reviewer manuscript, Supplementary Information, cover letter and anonymized code archive as separate files when the portal requests them.

## Portal files

- `output/pdf/manuscript.pdf`
- `output/pdf/supplementary_information.pdf`
- `output/pdf/cover_letter.pdf`
- `../dist/nature_machine_intelligence_supplementary_code.zip`
