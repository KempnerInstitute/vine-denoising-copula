# VDC 0.2.3

Separate software from the paper workspace. Paper-specific experiments, figures,
tables, saved results and historical cluster launchers now live in the companion
paper repository's `reproducibility/` directory. The library retains its training,
inference, data/model tools, examples, documentation and tests.

The paper-only checkpoint-discovery helper moved with its consumers. Model
implementations, public inference APIs, checkpoint weights and numerical defaults
are unchanged. Historical releases remain available under their original tags.
