# Version 0.2.5

- Corrected figure/table reproduction archive for the NeurIPS 2026 camera-ready paper
  (`vdc-neurips2026-figures-v0.2.5.zip`), following a final audit of the manuscript:
  - Figure 1(c) labels the fitted vine density hat-c(u); Figure 2(d) legend no longer covers the axis;
    Figure 4(c) and Figure A4 label placement; Figure A2 baseline label.
  - Figure A1 smoothing uses full-window moving averages (5,000 steps) instead of edge padding, which had
    created artificial rises at both ends of the curves.
  - Paired-replication and imputation tables are reformatted to fit the text width; two Table A26 ISE cells
    are rescaled before rounding (1.891, 2.715).
- No plotted value, saved input, runtime file (apart from the version string), checkpoint or published
  numerical result changes. Figures 3, A3, A5 and A6 are pixel-identical to v0.2.4.
