# ruff: noqa: E702  (verbatim port of the research implementation)
"""Continuous, reproducible pair-copula mixtures for training linear-grid models.

Each draw (:func:`draw_spec`) defines one population: an independence copula
(probability 0.05), a single parametric component (0.25) or a mixture of
2, 3, 4, 6 or 8 components (0.70; within one family with probability 0.15)
with Dirichlet weights of concentration 0.1, 0.3, 1 or 3. Components use the
Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1 and BB7 families, with
signs or rotations, and a component Kendall's tau drawn uniformly within one of
five strength bins covering 0.005-0.85. The same parameters define simulation,
density, CDF and conditional-CDF targets. BB6, BB8 and Tawn are not used.

The fine-tuning data of the ``vdc-denoiser-m64-v2`` and ``vdc-denoiser-m128-v2``
models were generated with this module's logic (pyvinecopulib 0.7.5). BB7
densities, CDFs and h-functions use stable log-domain evaluations because the
library returns NaN in some extreme tails.
"""
import numpy as np

try:
    import pyvinecopulib as pv
except ImportError as exc:
    raise ImportError(
        'Continuous-mixture training data requires pyvinecopulib: '
        'pip install "vine-denoising-copula[train]"'
    ) from exc

FAMILIES = ("gaussian", "student", "clayton", "gumbel", "frank", "joe", "bb1", "bb7")
STRENGTH_BINS = ((.005, .05), (.05, .2), (.2, .5), (.5, .7), (.7, .85))


def _bb7_parts(x, parameters, rotation=0):
    """Stable terms for BB7's generator and its inverse.

    Log-domain evaluation avoids cancellation of 1-(1-u)**theta in tails.
    Definition: vinecopulib/bicop/implementation/bb7.ipp, generator/inverse.
    This reference evaluator does not change the installed library.
    """
    uv = np.asarray(x, dtype=float).copy()
    if rotation in (90, 180):
        uv[:, 0] = 1-uv[:, 0]
    if rotation in (180, 270):
        uv[:, 1] = 1-uv[:, 1]
    if not np.all((uv > 0) & (uv < 1)):
        raise ValueError("BB7 density requires interior observations")
    theta, delta = np.asarray(parameters).ravel()
    z = theta*np.log1p(-uv)
    loga = np.empty_like(z)
    mask = z < -np.log(2)
    loga[mask] = np.log1p(-np.exp(z[mask]))
    loga[~mask] = np.log(-np.expm1(z[~mask]))
    terms = -delta*loga
    largest, smallest = terms.max(1), terms.min(1)
    logs = np.empty(len(uv))
    small = largest < 20
    logs[small] = np.log1p(np.expm1(terms[small]).sum(1))
    logs[~small] = largest[~small]+np.log1p(np.exp(smallest[~small]-largest[~small])-np.exp(-largest[~small]))
    logq = np.log(-np.expm1(-logs/delta))
    return uv, theta, delta, loga, logs, logq


def bb7_pdf_stable(x, parameters, rotation=0):
    """BB7 copula density at the rows of ``x`` (stable in the tails)."""
    uv, theta, delta, loga, logs, logq = _bb7_parts(x, parameters, rotation)
    first = (np.log1p(-1/theta)-np.log(delta)+(1/theta-2)*logq
             -(2/delta+2)*logs) if theta > 1 else np.full(len(uv), -np.inf)
    second = np.log1p(1/delta)+(1/theta-1)*logq-(1/delta+2)*logs
    logprefactor = (np.log(delta*theta)+(theta-1)*np.log1p(-uv).sum(1)
                    -(delta+1)*loga.sum(1))
    return np.exp(logprefactor+np.logaddexp(first, second))


def bb7_cdf_stable(x, parameters, rotation=0):
    """BB7 copula CDF at the rows of ``x``."""
    _, theta, _, _, _, logq = _bb7_parts(x, parameters, rotation)
    base = -np.expm1(logq/theta)
    if rotation == 90:
        return x[:, 1]-base
    if rotation == 180:
        return x.sum(1)-1+base
    if rotation == 270:
        return x[:, 0]-base
    return base


def bb7_h_stable(x, parameters, rotation=0):
    """BB7 conditional CDF ``P(U <= u | V = v)`` (pyvinecopulib's ``hfunc2``)."""
    uv, theta, delta, loga, logs, logq = _bb7_parts(x, parameters, rotation)
    logh = ((1/theta-1)*logq-(1/delta+1)*logs
            +(theta-1)*np.log1p(-uv[:, 1])-(delta+1)*loga[:, 1])
    return -np.expm1(logh) if rotation in (90, 180) else np.exp(logh)


def component(rng, family=None):
    """Draw one parametric component (family, parameters, rotation, strength bin)."""
    family = family or str(rng.choice(FAMILIES))
    strength_bin = int(rng.integers(len(STRENGTH_BINS)))
    low, high = STRENGTH_BINS[strength_bin]
    tau = float(rng.uniform(low, high))
    rotation = 0
    if family in ("gaussian", "student"):
        rho = float(rng.choice([-1, 1])) * np.sin(np.pi*tau/2)
        params = [[rho]] if family == "gaussian" else [[rho], [float(np.exp(rng.uniform(np.log(2.1), np.log(40.))))]]
        bicop = pv.Bicop(family=getattr(pv.BicopFamily, family), parameters=np.array(params))
    elif family in ("bb1", "bb7"):
        # Reject by achieved strength; uniform parameter ranges do not balance tau.
        for attempt in range(10000):
            if family == "bb1":
                params = [[float(np.exp(rng.uniform(np.log(.001), np.log(7.))))],
                          [1+float(np.exp(rng.uniform(np.log(.001), np.log(6.))))]]
            else:
                params = [[1+float(np.exp(rng.uniform(np.log(.001), np.log(5.))))],
                          [float(np.exp(rng.uniform(np.log(.01), np.log(25.))))]]
            bicop = pv.Bicop(family=getattr(pv.BicopFamily, family), parameters=np.array(params))
            if low <= bicop.tau <= high:
                break
        else:
            raise RuntimeError(f"Could not cover {family} strength bin {strength_bin}")
        rotation = int(rng.choice([0, 90, 180, 270]))
        bicop.rotation = rotation
    else:
        bicop = pv.Bicop(family=getattr(pv.BicopFamily, family))
        if family == "frank":
            tau *= int(rng.choice([-1, 1]))
        bicop.parameters = bicop.tau_to_parameters(tau)
        if family != "frank":
            rotation = int(rng.choice([0, 90, 180, 270]))
            bicop.rotation = rotation
        params = bicop.parameters.tolist()
    return {"family": family, "parameters": params, "rotation": rotation,
            "strength_bin": strength_bin, "component_tau": float(bicop.tau)}


def draw_spec(rng, mixture_probability=.7, independence_probability=.05):
    """Draw one population specification from ``numpy.random.Generator`` ``rng``."""
    choice = rng.random()
    if choice < independence_probability:
        comps = [{"family": "indep", "parameters": [], "rotation": 0,
                  "strength_bin": None, "component_tau": 0., "weight": 1.}]
        regime, alpha = "independence", None
    elif choice < independence_probability + mixture_probability:
        count = int(rng.choice([2, 3, 4, 6, 8]))
        within_family = bool(rng.random() < .15)
        families = [str(rng.choice(FAMILIES))]*count if within_family else rng.choice(FAMILIES, count, replace=False)
        comps = [component(rng, str(f)) for f in families]
        alpha = float(rng.choice([.1, .3, 1., 3.]))
        weights = rng.dirichlet(np.full(count, alpha))
        for comp, weight in zip(comps, weights):
            comp["weight"] = float(weight)
        regime = "within_family_mixture" if within_family else "between_family_mixture"
    else:
        comps = [dict(component(rng), weight=1.)]
        regime, alpha = "single", None
    return {"regime": regime, "dirichlet_alpha": alpha, "components": comps,
            "effective_components": float(1/sum(c["weight"]**2 for c in comps))}


class ContinuousMixture:
    """Mixture copula of a :func:`draw_spec` specification.

    ``simulate(n, seed)`` draws component assignments and component samples
    from ``numpy.random.default_rng(seed)``; ``pdf``, ``cdf`` and
    ``h_u_given_v`` evaluate the same mixture; ``cell_masses(edges)`` returns
    exact cell probabilities from the CDF on a rectilinear grid.
    """

    def __init__(self, spec):
        self.spec = spec
        self.weights = np.array([c["weight"] for c in spec["components"]])
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1, atol=1e-12):
            raise ValueError("Invalid mixture weights")
        self.copulas = []
        for c in spec["components"]:
            options = {"family": getattr(pv.BicopFamily, c["family"]), "rotation": c["rotation"]}
            if c["parameters"]:
                options["parameters"] = np.asarray(c["parameters"])
            self.copulas.append(pv.Bicop(**options))

    def pdf(self, x):
        values = []
        for w, c in zip(self.weights, self.copulas):
            pdf = (bb7_pdf_stable(x, c.parameters, c.rotation)
                   if c.family == pv.BicopFamily.bb7 else c.pdf(x))
            if not np.isfinite(pdf).all() or (pdf < 0).any():
                raise ValueError(f"Invalid reference PDF for {c}")
            values.append(w*pdf)
        return sum(values)

    def cdf(self, x):
        return sum(w*(bb7_cdf_stable(x, c.parameters, c.rotation)
                      if c.family == pv.BicopFamily.bb7 else c.cdf(x))
                   for w, c in zip(self.weights, self.copulas))

    def h_u_given_v(self, x):
        return sum(w*(bb7_h_stable(x, c.parameters, c.rotation)
                      if c.family == pv.BicopFamily.bb7 else c.hfunc2(x))
                   for w, c in zip(self.weights, self.copulas))

    def simulate(self, n, seed):
        rng = np.random.default_rng(seed)
        assignments = rng.choice(len(self.weights), size=n, p=self.weights)
        result = np.empty((n, 2))
        for k, copula in enumerate(self.copulas):
            mask = assignments == k
            if mask.any():
                result[mask] = copula.simulate(int(mask.sum()), seeds=[int(rng.integers(1, 2**31-1))])
        return result

    def cell_masses(self, edges):
        a, b = np.meshgrid(edges, edges, indexing="ij")
        cdf = self.cdf(np.clip(np.c_[a.ravel(), b.ravel()], 1e-12, 1-1e-12)).reshape(len(edges), len(edges))
        cdf[0, :] = 0; cdf[:, 0] = 0; cdf[-1, :] = edges; cdf[:, -1] = edges
        mass = np.diff(np.diff(cdf, axis=0), axis=1)
        if not np.isfinite(mass).all() or mass.min() < -1e-9:
            raise ValueError(f"Invalid numerical CDF masses: {mass.min()}")
        mass = np.maximum(mass, 0.)
        if max(np.max(np.abs(mass.sum(0)-np.diff(edges))), np.max(np.abs(mass.sum(1)-np.diff(edges)))) > 1e-8:
            raise ValueError("CDF-derived masses do not have the intended margins")
        return mass / mass.sum()
