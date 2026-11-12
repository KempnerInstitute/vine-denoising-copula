# Synthetic training data

Training examples pair samples with a density evaluated on a uniform midpoint
grid. The grid is normalized by its midpoint integral; this discretization is
an approximation to the continuous copula, particularly near strong tail peaks.
A mixture target is the weighted sum of its component grids. Component sample
counts are drawn from the same mixture weights.

The maintained generator supports Gaussian, Student-t, Clayton, Gumbel, Frank,
Joe, BB1, BB7 and independence copulas, including rotations and mixtures. Complex
grid-defined copulas are also available as individual examples. Joe, BB1 and BB7
sampling and BB-family density evaluation use pyvinecopulib 0.7.5. Tests compare
sample margins and joint probabilities with closed-form CDFs, and densities with
mixed derivatives of those CDFs.

The `conditional_*` options in historical configurations are rejected. They
paired a fixed-condition inverse-CDF sample with its own uniform driver, so the
two columns did not form a bivariate copula and did not match the density target.
Use ordinary copula families for a new training run. Higher-tree copulas also
have uniform conditional margins; fixing one argument of an h-function is not a
substitute for generating such a pair.

These corrections affect new training and generated evaluation data. They do
not modify the released checkpoint, its hash, or inference. Historical training
streams require their original source pin. Paper experiment instructions retain
those pins; running an old configuration with the corrected generator is not an
exact replay of the original training run.

The standalone conditional CDF and inverse-CDF functions remain available for
conditional diagnostics. Unsupported families and invalid training targets raise
errors rather than silently substituting an independence target.
