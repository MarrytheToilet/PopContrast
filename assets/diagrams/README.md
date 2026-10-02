# PopContrast overview

`popcontrast_overview.pdf` is the author's revised vector artwork for manuscript
Figure 2, supplied on 2026-10-01. `popcontrast_overview.png` is its supplied
preview. The PDF is the fixed LaTeX build input. The historical imagegen edit
prompt records the earlier layout; it does not regenerate this revised artwork.

The shared publication builder copies both files byte for byte. LaTeX scales
the PDF to the full two-column width, preserving its vector lettering and
formulas. The source manifest records both checksums, native and print
dimensions, preview resolution, and effective vector font sizes.

The upper route estimates the default geometric reference from complete-SID
log-scores over shared sampled training histories, then standardizes across
catalog items. The lower route uses the same trained recommender and corrects
completed candidates. The illustration moves B from rank 3 to rank 2 across a
K=2 cutoff while retaining the candidate set. Icons, heatmap intensities and
ranking changes are schematic. Validation selection happens before deployment;
the 95% retention constraint is measured on validation, not a test guarantee.

English and Chinese captions and accessibility descriptions live with their
respective manuscript sources. Superseded imagegen drafts have been removed;
this directory holds the production artwork and its final edit prompt.
