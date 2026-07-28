# Nature Machine Intelligence submission edition

This directory is a separate, double-anonymized Nature Machine Intelligence edition. It does not replace or modify the TMLR manuscript.

## Files

- `manuscript.tex`: Article manuscript, including Methods and editorial declarations.
- `supplementary_information.tex`: detailed controls, downstream statistics, hashes and reproducibility information.
- `cover_letter.tex`: editorial cover letter with explicit author-only placeholders.
- `nature_references.bib`: compact bibliography used by the Nature manuscript.
- `output/pdf/`: compiled PDFs after a successful build.

## Build

From this directory:

```bash
mkdir -p output/pdf output/build/manuscript output/build/supplement output/build/cover
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/build/manuscript manuscript.tex
bibtex output/build/manuscript/manuscript
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/build/manuscript manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/build/manuscript manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/build/supplement supplementary_information.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/build/cover cover_letter.tex
cp output/build/manuscript/manuscript.pdf output/pdf/
cp output/build/supplement/supplementary_information.pdf output/pdf/
cp output/build/cover/cover_letter.pdf output/pdf/
```

## Author actions before submission

1. Replace every `[AUTHOR ACTION REQUIRED: ...]` field in the cover letter.
2. Add the final author contribution and competing-interest statements to the manuscript after choosing single- or double-anonymized review.
3. Confirm that the overlapping TMLR submission has been withdrawn or has reached a final decision. Do not submit overlapping manuscripts concurrently.
4. Confirm data and code URLs once an anonymous review deposit or public archive is available.
5. Recheck the current Nature Machine Intelligence instructions immediately before upload.

## Format checks

The source is designed as an initial Article submission: an abstract below 150 words, unheaded introduction, Results, Discussion and Methods, and no more than six main display items. Nature Machine Intelligence does not require a journal-specific LaTeX template for initial submission.
