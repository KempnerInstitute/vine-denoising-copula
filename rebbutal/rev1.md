Meta Review of Submission33494 by Area Chair Htv7
Meta Reviewby Area Chair Htv720 Jul 2026, 23:57 (modified: 23 Jul 2026, 14:36)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair Htv7Revisions
Metareview:
The paper presents Vine Denoising Copula(VDC) which trains a single neural network and reuses it to alleviate the cost of learning copulas. The reviewers feel that, though niche, it may be interesting to Copula Community. It justifies the claims through rigorous theorems and experimental results. There are some concerns about the clarity of presentation and experimental details.
One of the major concerns( see Reviewer gb3Z) is it not clear on what exactly is gained by introducing VDC. The authors note, line 83-87, flow based models are powerful but monolithic. However did not see any elaboration on why this is disadvantageous and how VDC fixes it. I find the comparison to Diffusion/Flow based models interesting. However it needs to be articulated better. Currently the assessment of the paper is mostly borderline. To make a stronger case for acceptance, the author(s) should try to address all concerns raised by the reviewers. I would encourage the authors to provide more evidence, if relevant, on how VDC can either augment or compete with Diffusion/Flow models,

Add:
Official Review of Submission33494 by Reviewer DzsK
Official Reviewby Reviewer DzsK12 Jul 2026, 22:55 (modified: 23 Jul 2026, 13:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer DzsKRevisions
Summary:
The article proposes Vine Diffusion Copulas (VDC), a new framework of single pre-trained neural edge estimator that replaces the repeated optimization of bivariate-copulas in classical vine copula models. Instead of fitting every bivariate copula separately, VDC converts each pair of variables into a 2D histogram, passes it through a U-Net denoiser to estimate the copula density, and then applies Iterative Proportional Fitting Procedure (IPFP) to enforce valid copula constraints before assembling the full vine using standard h-function recursion. This approach reduces fitting time while maintaining competitive density estimation accuracy. These explicit copula densities enable direct estimation of mutual information and total correlation. Experiments on synthetic copulas, UCI benchmark datasets, and S&P100 financial data show that VDC achieves good likelihood performance while fitting vines faster than existing vine methods and scaling effectively to hundreds of dimensions. The authors also acknowledge limitations under the simplifying vine assumption, continuous-valued data, fixed grid resolution, and conditional downstream tasks such as Value-at-Risk prediction and missing-data imputation, where calibration errors can accumulate through the vine recursion.

Contribution Type: Use-inspired: The main contribution is in framing or designing approaches to meet the needs of a specific real-world application. (This often involves, e.g., engaging with domain experts.)
Strengths And Weaknesses:
Strengths

Accurate mutual information estimation: VDC accurately estimates edge-wise mutual information and total correlation on synthetic Clayton-vine benchmarks.

Fast pair-copula estimation: For copula families seen during training, VDC achieves low Integrated Squared Error (ISE) while requiring only a few milliseconds per edge.

Strong performance on synthetic benchmarks: On the homogeneous Clayton D-vine benchmark, VDC achieves competitive held-out negative log-likelihood (NLL) with substantially lower fitting time than classical vine methods.

Competitive real-data results: On several UCI datasets, VDC achieves competitive held-out NLL (bits/dim) while fitting significantly faster than strong nonparametric vine baselines.

Weaknesses

Limited theoretical contribution: The paper provides almost no theoretical analysis beyond the standard IPFP/Sinkhorn convergence result. It does not establish statistical consistency, convergence rates, approximation error, bias, or sample complexity of the proposed estimator.

Difficult to read: The manuscript relies heavily on specialized terminology from copula theory, diffusion models, and information theory, making it difficult for readers outside these areas to follow. Moreover, the inconsistent organization of figures and tables between the main text and the appendix makes the paper difficult to follow.

Missing mathematical definitions: Many mathematical objects and experimental quantities are introduced without precise definitions, reducing clarity.

Incomplete experimental details: The experimental protocol omits important details, including the composition of the held-out synthetic benchmarks, the number of evaluation instances, averaging procedures, and parts of the benchmark generation process. Improper experiential design: Several numerical experiments rely on a very small number of independent runs (typically 3–5). Substantially more Monte Carlo runs (e.g., 100–1000, when computationally feasible) would provide more robust estimates. For example, the standard deviations reported in Table A7 are relatively large compared to the corresponding mean values. Similarly, the self-consistency experiments in Appendix D.7 are based on only five trials per configuration. Furthermore, the choice of the tolerance parameter 
 in the DPI test is introduced without theoretical or empirical justification, making it unclear how sensitive the reported violation rates are to this threshold.

Quality: 2: not good
Clarity: 2: not good
Significance: 2: not good
Originality: 4: excellent
Questions:
Suggestions:

Improve the organization and cross-referencing of experimental results: The mapping between claims in the main text and the supporting appendix material should be made direct and explicit. For example, Figure 5 is supported by multiple appendix tables (Tables A5, A14, and A15) located in different sections, yet the figure caption does not refer readers to these tables, making it difficult to locate the detailed numerical results. Similarly, Table A7 provides the detailed version of the benchmark summarized in Table 1, but this relationship is never explicitly stated, leaving readers to infer the connection.

Provide precise mathematical definitions and benchmark specifications: Several mathematical quantities and technical terms are introduced without adequate definitions. For example, the variables 
 and 
 are not formally defined when first introduced, and the function 
 is used without a precise definition. Likewise, the "copula zoo" is only described qualitatively; the parameter ranges for each copula family, the construction of the "complex synthetic patterns," and the procedure for generating mixture copulas are omitted. The held-out synthetic benchmark is also insufficiently specified: the paper does not report the number of held-out copulas, their parameter ranges, or the number of independent realizations used for evaluation. These details are essential for clarity and reproducibility.

Strengthen the experimental protocol and improve reproducibility: Several numerical experiments are based on only 3–5 independent runs, which provides limited evidence for the stability of the reported results. Although the appropriate number of repetitions depends on computational cost, substantially more runs (e.g., tens or hundreds whenever computationally feasible) would yield more reliable estimates. Furthermore, many experimental protocols are incompletely described in the manuscript. Important details such as the composition of the evaluation benchmarks, the number of repetitions, averaging procedures, and benchmark generation protocols are either missing or scattered throughout the appendix. Consequently, many of the reported experiments cannot be faithfully reproduced using the information provided in the paper alone.

Limitations:
Yes

Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No concern

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission33494 by Reviewer VcZn
Official Reviewby Reviewer VcZn26 Jun 2026, 08:02 (modified: 23 Jul 2026, 13:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer VcZnRevisions
Summary:
The paper introduces an amortized model for estimating simplified vine copulas, a class of multivariate models build exclusively via 2D copula building blocks. Authors pre-train a 2D histogram model on many synthetic bivariate copula examples, which is called once during inference, directly outputting the desired copula building block (as opposed to current approaches that need to fit each pair-copula). The amortized cost of fitting lends speedups, all the while achieving competitive performance on density estimation and mutual information or total correlation estimation.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
The method is an interesting application of pre-trained models to vine copulas. While the core topic of vine copula estimation is quite specific, the introduction of pre-trained models is a nice combination of ideas. The advantage is demonstrated in experiments. However the writing is quite unclear in the method and experiments Sections.

Strengths
Introducing a pretrained bivariate copula model in vines is sensible as the amortized speedup compounds with the quadratic number of pair-copulas needed with the dimension.

The method is effective at retaining comparable quality of density and MI/TC estimaiton while delivering substantial speedups.

I like the combination of the IPFP on top of a 2D histogram model to ensure the final model is a valid copula.

Weaknesses
The paper is very unclear in the Methods and Experiments Sections, and could benefit from rewriting/restructuring.
As this paper targets the niche of vine copula estimation, it's main audience are the statistical learning/probabilistic modelling subcommunities of NeurIPS. In my opinion, the following points should be better explained, either with a couple words in the main text, a citation, or a reference to an appropriate explanation in the appendix:

L114 U-Net should have a citation, maybe also a definition 

L 114 coordinate channels, do you mean dimension indexes? Give a methematical expression along the lines of 

L117 in eq 5, clip should be written as max(min(a,min(x,b)), also IPFP has not yet been discussed so the reader doesnlt know wehat it is

L118 Film should be cited

L119 output head should again have a mathematical function expression

L123 corrupt, give a reference to an appendix or give description on what that means, we don't know at this point in the text

L123-124 a scalar...density, either reference an appendix with thorough description or explain better here.

L124 default checkpoint, what do you mean?

L127 one-shot, for the first use of the term, explain this means in one function call

L130 131 output parameterisation, reference the equation to be precise

L131 IPFP is not referenced yet, so either don't mention it until later or explain it properly here

L 133 corruption ablation, we don;t know what that mean yet and there is no reference ot what it might be

L134-135, what does the sentence after : mean?

Eq 6 and L142-144 , what are each of the losses and how do you pick the lambdas? give expression or reference to them

Fig 2, the caption is not clear at this point since experiments have not been explained, and no winrate discussion has happened yet. Also, the equations in Fig1 should be present in the text when discussing the training variants, not just in the figure

Proposition 1 should be rewritten or the text before it, to better explain the IPFP procedure and how it is itterated, with possible mathematical details in the appendix?

L 164 stable projection regime, do you mean results in a similar fit regardless of K>15 so 15 is sufficient?

L168 the recursion, be more exact, do you mean the itterated pair copula construction of vines?

L169 assembly, is unclear, perhaps here pointing to an algorithm of your approach would help

L173-177 You already gave a similar overview of your approach before, perhaps use this space to instead include an algorithm and refernce it in the text? Then use 3.4 only for practical details

L176 histogramming?

L200-202 VaR related results are not referenced on mentioned before. Point to a proper description if you want to make a point about this experiment.

L 218 beating RealNVP does not happen across all datasets so the claim needs to be adjusted

Sec 5.3 is unclear with the descriptions of experiments and results alike

Fig 4 in the caption what is meant by the seeds 
?

L 253-254 what is meant by weak residual dependence and large-recursion blow up?

L261-262 fixed-checkpoint... sensitivity, what does this mean?

Fig 5 caption, same comment on seeds, also what is meant by suite in "suite means over copula families"?

L279 again there is no proper detail on what this VaR experiment is

Aren't the sample sizes in 5.5 are very high for 2D Mi estimation. Similar for Fig 4? I would think these are typically a bit outside the application areas of vines? Also, for Fig 4, do vines not scale with sample size in the non-parametric TLL version, meaning their computational time goes up with larger n?

could you include experiments on sampling? Or does your method not support it easily?

Minor points:

L79: Tagasovska et al 2019 is not a global neural copula model, rather it uses a vine in the latent space of a VAE, so it doesn't quite fit in this section discussing deep copulas.

L74 Aren't there other works on deep archimedean copulas?

Quality: 3: good
Clarity: 1: poor
Significance: 3: good
Originality: 3: good
Questions:
I generally like the paper as it is a relevant application of pretrained models for vines. However the clarity is not at a NeurIPS standard in my option, and is not appropriate for the target audience that would most benefit from this paper either. I would like to see suggestions on practically most of the comments on clarity before I can confidently advocate for the acceptance of this work. It would also be nice to see if sampling is possible and competitive.

Other questions that aren't as relevant:

Can this pre-training approach of histograms followed by IPFP be used on d>2? If not, what is the bottleneck here?
Limitations:
yes

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission33494 by Reviewer gb3Z
Official Reviewby Reviewer gb3Z25 Jun 2026, 18:49 (modified: 23 Jul 2026, 13:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer gb3ZRevisions
Summary:
This paper proposes an amortized method for fitting simplified vine copulas. Instead of fitting each pair-copula edge separately, VDC trains one reusable bivariate neural estimator on synthetic copula examples and applies it to all vine edges. Each edge density is predicted on a grid, projected to satisfy copula marginal constraints, and used to compute h-functions for standard vine recursion. The main contribution is a faster, flexible edge-fitting procedure that preserves the interpretability and likelihood structure of classical vines while enabling efficient density and information estimation.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
The main strength of the paper is that it introduces an amortized inference approach for simplified vine copulas. Instead of fitting each pair-copula edge separately, the method pretrains a neural network that maps pseudo-observations, represented as empirical copula histograms, to discretized bivariate density estimates. The paper also incorporates projection steps to enforce approximately uniform copula marginals and computes h-functions from the estimated density grids for standard vine recursion.

The main weakness is that the paper does not clearly explain what is gained and what is lost relative to standard inference methods. The significance of the amortized approach should be justified: does it enable tasks that standard pair-copula estimation cannot achieve, or mainly provide a faster approximation? More details come in the following section.

Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
The name “Vine Denoising Copula” may be somewhat misleading. The proposed method is not a new class of vine copula model, but rather an amortized estimation procedure for the pair-copula.

The main idea is that the method learns a general-purpose machine for bivariate copula density estimation. However, standard nonparametric copula density estimators already serve this role and have more established theoretical properties. The paper should more clearly justify why amortized neural estimation is preferable to simply applying a nonparametric estimator to each edge.

The fast inference seems to come at a cost. Ultimately, a model should provide a probabilistic description of the data that supports interpretation, simulation, prediction, and dependence analysis. However, the proposed method outputs discretized pair-copula density grids. It is therefore unclear how complete or reliable the resulting probabilistic description is. Can the fitted model be used to generate realistic new datasets? Can it support accurate conditional prediction or imputation? Can it recover meaningful dependence properties, especially tail dependence and higher-tree conditional dependence?

The paper should also discuss the reliability of the pretrained estimator, including how users can diagnose when it fails on dependence structures that differ from the synthetic training copulas.

Limitations:
Please refer to the comments above.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission33494 by Reviewer LCT8
Official Reviewby Reviewer LCT817 Jun 2026, 19:20 (modified: 23 Jul 2026, 13:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LCT8Revisions
Summary:
The paper proposes Vine Denoising Copula (VDC), an amortized pipeline for fitting simplified vine copulas to continuous data. Classical vines fit a separate bivariate copula on every edge, which is the cost bottleneck as dimension grows. VDC instead trains one bivariate denoising U-Net on a synthetic copula zoo, then reuses the frozen network on every edge. For each edge it histograms the pseudo-observations, predicts a log-density grid in one forward pass, projects the grid to uniform marginals with an IPFP/Sinkhorn step, and caches h-functions by cumulative sums for the next tree level. Because the densities are explicit, mutual information and total correlation follow from copula-entropy identities with edge-wise decomposition. Experiments cover held-out synthetic bivariate copulas, UCI and S&P100 full-vine density, high-dimensional Clayton-vine scaling to d=500, MI and TC accuracy against KSG and neural baselines, several ablations, and two honest negative results on VaR and imputation. The headline claims are large bivariate accuracy gains, competitive full-vine likelihood, faster high-dimensional fitting once the one-time pretraining is amortized, and self-consistent information estimates, with conditional downstream tasks flagged as the main limitation.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
The formal content is correct where checked. The h-function identities in Equation (3) and Appendix B.2 are right, since for a copula the conditional density of U given V equals c(u,v). The mutual-information identity and the total-correlation decomposition under a vine factorization are stated correctly. Proposition 1 is the standard Sinkhorn result, and the strict-positivity hypothesis genuinely holds because the network output is a clipped exponential. The engineering is clean. Amortizing only the bivariate edge while keeping classical vine recursion, then repairing marginals after the forward pass with IPFP, is a sensible and reusable design that preserves explicit pair-copulas and h-functions. The experimental suite is broad and unusually honest, with full ablations on corruption, IPFP iterations, probit transforms, and flow capacity, and with reported failures on imputation and VaR rather than hidden ones. There are real density wins where dependence is genuinely non-Gaussian, namely the synthetic Clayton vine and the S&P100 returns.

The first and most load-bearing concern is that the headline accuracy is largely in-distribution. The synthetic test copulas in Table 1 are drawn from the same families used to train the network, so the roughly hundredfold ISE advantage over pyvine-TLL is measured on the network's own training distribution. The hard synthetic families in Table A11, such as ring and double-banana, push partly outside this set and VDC still wins, which is encouraging, but those shapes plausibly overlap the complex synthetic patterns in the training zoo, so this is not a clean out-of-distribution test. The paper should state plainly that the synthetic numbers establish in-distribution interpolation accuracy, and should add a test on copula families or dependence structures explicitly excluded from training.

The second concern is that this bivariate accuracy does not transfer to real-data density. On every UCI dataset in Table A1, pyvine-TLL matches or beats VDC on held-out NLL. On Hepmass, Credit, and Miniboone the copula NLL sits at essentially zero bits per dim for pyvine, VDC is slightly positive, which is worse than independence, and a single-covariance Gaussian copula also matches or beats VDC. A positive copula NLL means the fitted dependence does slightly worse than independence on held-out data. So on these sets there is little dependence signal, both strong baselines hug the independence floor, and VDC adds a small amount of structure that does not help and sometimes hurts. This is a coherent story, since the real wins come on data with non-Gaussian or tail dependence, but it is narrower than the abstract's framing, and the paper should be clear that several UCI benchmarks carry too little dependence to discriminate density methods.

Third, the speed advantage is confounded with hardware. Appendix C.7 disclose that VDC runs on an H100 GPU while the baselines are pinned to a single CPU thread. vinecopulib is a heavily optimized multithreaded C++ library, so comparing a full GPU against one CPU core conflates algorithmic amortization with GPU acceleration. The disclosure is honest and the GPU-amortized regime is a legitimate deployment target, but every speedup inherits this caveat, so the headline efficiency claim cannot be read as a property of the method until a matched-compute comparison is provided, either multithreaded vinecopulib or VDC inference on CPU. The 18-hour pretraining and the tens-to-hundreds-of-vines break-even further restrict the regime in which the method pays off, which the paper does state.

Fourth, the self-consistency advantage over KSG is thin. The DPI result is 0 out of 60 for VDC versus 2 out of 60 for KSG, and the two Wilson intervals overlap, at 0 to 6.0 percent and 0.9 to 11.4 percent. The additivity gap is 0.021 versus 0.027 nats. The additivity test also computes a four-dimensional MI for the left-hand side, where KSG is known to degrade with dimension, as the paper's own TC scaling shows KSG underestimating by about twofold at 
. So part of the apparent edge reflects KSG's dimensional bias rather than a coherence advantage of VDC. The claim should be softened to a marginal, possibly not significant, difference.

Fifth, two representation concerns interact with the acknowledged conditional failures. The grid is 64 by 64, and the probit ablation in Table A17 shows that tail and interior behavior shift with resolution, so tail dependence, h-functions, and inverse h-functions may be sensitive to grid resolution, smoothing, and projection. IPFP fixes marginal validity but does not guarantee that chained conditional inference stays calibrated, and it could in principle distort sharp or tail structure while repairing marginals. Separately, the method estimates one pooled conditional pair-copula per edge, so it cannot represent conditional copulas that vary with the realized conditioning value. This pooled-conditional limitation is genuine, but it is shared with classical simplified vines such as pyvine-TLL, which do not show the same conditional breakdown, so it is not the specific cause of VDC's weak VaR and imputation results. The paper's own diagnosis, that small edge-calibration errors accumulate through h and inverse-h transforms, is the more likely cause, and the deep-tree evidence bears on this. Table A12 reads near-zero deep-edge NLL as weak residual dependence, but near-zero NLL is observationally identical to an estimator defaulting toward independence as discretization and recursion error accumulate. A controlled synthetic deep vine with known nonzero deep-tree dependence would separate these explanations and would speak directly to the conditional-task failures.

On originality, the components are individually known. Neural copulas, Sinkhorn projection, and the MI-equals-copula-entropy identity all predate this work and are cited. The novelty is the specific combination of a single amortized bivariate edge operator with post-hoc IPFP and classical vine recursion. That is a useful and meaningful new pipeline rather than a new modeling idea.

On clarity, the paper is well written and the pipeline is easy to follow. The main text should state that the large synthetic numbers are in-distribution, that the UCI density comparison sits near the independence baseline, and that the speedups assume GPU amortization and repeated use. Several figures are too small to read.

A few positioning claims rest on external baselines I cannot verify, including ACNet, the Huk classifier and diffusion copulas, and MINDE. I flag these as unverified, though they appear standard.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
The training zoo and the synthetic test families overlap. How much of the roughly hundredfold bivariate ISE advantage reflects in-distribution family coverage rather than generalization? Please add a held-out test on copula families or structures explicitly excluded from training.

All reported speedups compare H100 GPU VDC to single-thread CPU baselines. Can you provide a matched-compute comparison, either multithreaded vinecopulib or VDC on CPU, to separate algorithmic amortization from GPU acceleration?

On Hepmass, Credit, and Miniboone, pyvine-TLL and even a single-covariance Gaussian copula match or beat VDC, and copula NLL sits near zero bits per dim. Do these datasets carry enough dependence to discriminate density methods, and is VDC's small positive NLL evidence of mildly harmful added structure?

The DPI result is 0 out of 60 versus 2 out of 60 with overlapping Wilson intervals, and additivity is 0.021 versus 0.027 nats. Is the self-consistency advantage over KSG statistically meaningful, and does the four-dimensional additivity test disadvantage KSG through its known high-dimensional bias?

Can you give a sharper diagnosis of the conditional-task failures, separating edge-density error, h-function discretization, inverse-h calibration, empirical marginal inversion, and accumulated recursion error, and does IPFP ever distort tail or sharp dependence while repairing marginals? A controlled deep synthetic vine with known nonzero deep-tree dependence would help.

Limitations:
Section 6 and the appendices are candid about the simplifying assumption, fixed grid resolution, finite IPFP tolerance, the pretraining amortization regime, and especially the conditional-calibration boundary, with honest negative results for imputation and VaR. Four gaps should be added. The headline bivariate accuracy is in-distribution. The timing comparison is GPU against single-thread CPU. On the weak-dependence UCI sets a trivial Gaussian copula matches or beats VDC, so competitive density there is near a trivial baseline. The self-consistency advantage over KSG is small with overlapping confidence intervals.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
N/A

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes