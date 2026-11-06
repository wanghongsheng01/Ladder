"""Ladder: cost-optimal approximate semantic aggregation over a catalog of LLM labelers.

Notation follows the paper. Population R (N tuples); oracle value Y(t) (indicator for COUNT, rating for AVG);
labelers k = 1..K with per-tuple cost c_k and output L_k(t) in {-1, 0, ..., C-1}.
A *surrogate* s is a set (bundle) of labelers; its value f_s(t) is a linear calibration of the one-hot outputs
of its labelers, fitted on the pilot sample. A *ladder* is a chain of surrogates s_1, ..., s_J ordered by decreasing
rho^2 = corr(Y, f_s)^2. Nested samples S_0 (oracle + all) within S_1 within ... within S_J are drawn uniformly
without replacement. The estimator is
    theta = mean_{S_0}(Y) + sum_j a_j * (mean_{S_j}(f_j) - mean_{S_{j-1}}(f_j)).
"""
import itertools
import numpy as np
from scipy.optimize import minimize

Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


# ---------------------------------------------------------------- surrogates
def onehot(preds, C):
    """preds: (n, k) int array with values in -1..C-1 -> (n, k*(C+1)) one-hot (a slot for 'unparsed')."""
    n, k = preds.shape
    X = np.zeros((n, k * (C + 1)))
    for j in range(k):
        X[np.arange(n), j * (C + 1) + preds[:, j] + 1] = 1.0
    return X


class Surrogate:
    """Ridge-calibrated linear surrogate on the one-hot outputs of a bundle of labelers."""

    def __init__(self, members, C, lam=1.0):
        self.members, self.C, self.lam = tuple(members), C, lam

    def fit(self, P, y):
        X = onehot(P[:, self.members], self.C)
        X = np.hstack([np.ones((len(X), 1)), X])
        R = self.lam * np.eye(X.shape[1]); R[0, 0] = 0
        self.w = np.linalg.solve(X.T @ X + R, X.T @ y)
        return self

    def __call__(self, P):
        X = onehot(P[:, self.members], self.C)
        return self.w[0] + X @ self.w[1:]


def cv_rho2(P, y, members, C, folds=5, rng=None):
    """Cross-validated squared correlation between y and the surrogate (avoids optimism on small pilots)."""
    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(y)); f = np.empty(len(y))
    for part in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, part)
        f[part] = Surrogate(members, C).fit(P[tr], y[tr])(P[part])
    if np.std(f) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return float(np.corrcoef(y, f)[0, 1] ** 2)


# ---------------------------------------------------------------- variance model
def variance(u, rho2, sig2, Ninv):
    """Var of the ladder estimator with optimal coefficients. u = (1/m_0, ..., 1/m_J); rho2 = (rho_1^2..rho_J^2)."""
    v = u[0] - Ninv
    for j in range(1, len(u)):
        v -= (u[j - 1] - u[j]) * rho2[j - 1]
    return sig2 * v


def closed_form(costs, rho2, c0, sig2, V, N):
    """MFMC-style optimal sample sizes for target variance V (ignoring m <= N); returns m (len J+1) and cost."""
    c = np.r_[c0, costs]; r = np.r_[1.0, rho2, 0.0]
    d = r[:-1] - r[1:]
    if np.any(d <= 0): return None, np.inf
    ratio = np.sqrt(d / c)
    # nesting m_0 <= m_1 <= ... requires d_j/c_j increasing
    if np.any(np.diff(ratio) < -1e-12): return None, np.inf
    S = np.sum(np.sqrt(c * d))
    target = V / sig2 + 1.0 / N
    if target <= 0: return None, np.inf
    m = ratio * S / target
    return m, float(np.dot(c, m))


def optimise(costs, rho2, c0, sig2, V, N, m_min):
    """Minimise expected cost subject to Var <= V, m_min <= m_0 <= m_1 <= ... <= m_J <= N. Uses u = 1/m."""
    c = np.maximum(np.r_[c0, costs], 1e-9); J = len(costs)
    m0, cost0 = closed_form(costs, rho2, c0, sig2, V, N)
    x0 = np.clip(m0 if m0 is not None else np.full(J + 1, N / 2), m_min, N)
    x0 = np.maximum.accumulate(x0)
    lo, hi = np.log(m_min), np.log(N)
    obj = lambda z: float(np.dot(c, np.exp(z)))
    cons = [{"type": "ineq", "fun": lambda z: V - variance(np.exp(-z), rho2, sig2, 1.0 / N)}]
    cons += [{"type": "ineq", "fun": (lambda z, j=j: z[j + 1] - z[j])} for j in range(J)]
    res = minimize(obj, np.log(x0), method="SLSQP", bounds=[(lo, hi)] * (J + 1), constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-12})
    m = np.exp(res.x)
    if variance(1 / m, rho2, sig2, 1 / N) > V * 1.001:  # infeasible: everything at N except oracle
        return None, np.inf
    return m, float(np.dot(c, m))


# ---------------------------------------------------------------- ladder selection
def effective_costs(chain, unit):
    """Labeler k is run on the largest sample of a surrogate that contains it, so surrogate j only pays for
    labelers that no deeper surrogate uses. chain: list of member tuples; unit: dict labeler -> cost."""
    out = []
    for j, mem in enumerate(chain):
        later = set().union(*chain[j + 1:]) if j + 1 < len(chain) else set()
        out.append(sum(unit[k] for k in mem if k not in later))
    return out


def chain_score(chain, rho2, unit, c0):
    """sum_j sqrt(c~_j (rho_j^2 - rho_{j+1}^2)) including the oracle level; inf if the nesting condition fails.
    The minimal cost for a target variance V is sig2 * score^2 / (V/sig2 + 1/N) (Theorem 2)."""
    # a free labeler (an existing column) gets a tiny positive cost; the allocation then pushes its sample to N
    c = np.maximum(np.array([c0] + effective_costs(chain, unit), float), 1e-9)
    r = np.array([1.0] + list(rho2) + [0.0])
    d = r[:-1] - r[1:]
    if np.any(d <= 1e-9): return np.inf
    ratio = np.sqrt(d / c)
    if np.any(np.diff(ratio) < -1e-12): return np.inf
    return float(np.sum(np.sqrt(c * d)))


def skyline(cands):
    """Keep surrogates not dominated in (cost, rho2): Pareto frontier, cheaper or more informative."""
    keep = []
    for i, (m, c, r) in enumerate(cands):
        if not any((c2 <= c and r2 >= r) and (c2 < c or r2 > r) for j, (m2, c2, r2) in enumerate(cands) if j != i):
            keep.append(i)
    return keep


def best_chain(cands, unit, c0, max_len=3):
    """cands: list of (members, cost, rho2). Enumerate chains of skyline surrogates (sorted by rho2 desc) up to
    max_len and return the one with the smallest score (oracle-only has score sqrt(c0))."""
    sky = sorted(skyline(cands), key=lambda i: -cands[i][2])
    best = (np.sqrt(c0), [])
    for L in range(1, max_len + 1):
        for path in itertools.combinations(sky, L):  # combinations keep the rho2-descending order
            sc = chain_score([cands[i][0] for i in path], [cands[i][2] for i in path], unit, c0)
            if sc < best[0]: best = (sc, list(path))
    return best[1], best[0]


# ---------------------------------------------------------------- execution
def nested_samples(pool, m, rng):
    """pool: candidate tuple ids; m: increasing sizes. Returns list of nested id arrays S_0 within ... within S_J."""
    perm = rng.permutation(pool)
    sizes = np.minimum(np.ceil(m).astype(int), len(pool))
    sizes = np.maximum.accumulate(np.maximum(sizes, 2))
    return [perm[:s] for s in sizes]


def estimate(Y, F, samples, alphas, N_rest, conf=0.95):
    """Ladder estimate of the mean of Y over the non-pilot population, with a CLT confidence interval.
    Y: oracle values (only read on samples[0]); F: list of surrogate value arrays over the population."""
    S0 = samples[0]
    th = Y[S0].mean()
    for j, (f, a) in enumerate(zip(F, alphas), start=1):
        th += a * (f[samples[j]].mean() - f[samples[j - 1]].mean())
    # plug-in variance with fixed alphas (estimated on the pilot)
    u = [1.0 / len(s) for s in samples]
    var = np.var(Y[S0], ddof=1) * (u[0] - 1.0 / N_rest)
    for j, (f, a) in enumerate(zip(F, alphas), start=1):
        sf = np.var(f[samples[j]], ddof=1)
        cyf = np.cov(Y[S0], f[S0])[0, 1]
        var += (u[j - 1] - u[j]) * (a * a * sf - 2 * a * cyf)
    var = max(var, 1e-18)
    hw = Z[conf] * np.sqrt(var)
    return th, hw


def optimise_multi(costs, rho2, c0, sig2, V, N, m_min):
    """GROUP BY allocation: one shared ladder, G groups. rho2: (G, J); sig2, V: (G,).
    Minimise expected cost s.t. Var_g <= V_g for every group (each Var_g is linear in u = 1/m, so the
    feasible set is convex in u and the cost sum_j c_j / u_j is convex)."""
    c = np.r_[c0, costs]; J = len(costs); G = len(sig2)
    lo, hi = np.log(m_min), np.log(N)
    cons = [{"type": "ineq", "fun": (lambda z, g=g: (V[g] - variance(np.exp(-z), rho2[g], sig2[g], 1.0 / N)) / V[g])}
            for g in range(G)]
    cons += [{"type": "ineq", "fun": (lambda z, j=j: z[j + 1] - z[j])} for j in range(J)]
    best = None
    for start in [0.3, 0.6, 0.9]:
        x0 = np.linspace(lo + start * (hi - lo) * 0.5, lo + start * (hi - lo), J + 1)
        res = minimize(lambda z: float(np.dot(c, np.exp(z))), x0, method="SLSQP", bounds=[(lo, hi)] * (J + 1),
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        m = np.exp(res.x)
        ok = all(variance(1 / m, rho2[g], sig2[g], 1 / N) <= V[g] * 1.01 for g in range(G))
        if ok and (best is None or np.dot(c, m) < best[1]): best = (m, float(np.dot(c, m)))
    return best if best else (None, np.inf)


def estimate_prefix(Y, P, surrogates, samples, alphas, N_rest, conf=0.95):
    """Ladder estimate and CLT interval. Surrogate j is evaluated only on S_j (S_{j-1}, S_0 are prefixes of S_j).
    Variance in the residual form of Theorem 1: sum_j (u_j - u_{j+1}) Var(Y - a_{j+1} f_{j+1}), with f_{J+1} = 0 and
    u_{J+1} = 1/N; every term is a variance, so the plug-in estimate cannot cancel to zero."""
    S0 = samples[0]; m0 = len(S0)
    y0 = Y[S0]
    th = y0.mean(); u = [1.0 / len(s) for s in samples] + [1.0 / N_rest]
    F0 = []
    for j, (s, a) in enumerate(zip(surrogates, alphas), start=1):
        fj = s(P[samples[j]])
        th += a * (fj.mean() - fj[: len(samples[j - 1])].mean())
        F0.append(a * fj[:m0])
    var = 0.0
    for j in range(len(samples)):
        resid = y0 - F0[j] if j < len(F0) else y0
        var += (u[j] - u[j + 1]) * np.var(resid, ddof=1)
    return th, Z[conf] * np.sqrt(max(var, 0.0))


# ---------------------------------------------------------------- two-phase stratified plans (Neyman 1938)
def strata_from_surrogate(f_pilot, min_count=8):
    """Strata = cells of the surrogate (tuples with equal calibrated value), merged in value order until each
    has at least min_count pilot tuples. Returns sorted cut points on the surrogate value."""
    vals, cnt = np.unique(np.round(f_pilot, 9), return_counts=True)
    cuts, acc = [], 0
    for v, c in zip(vals, cnt):
        acc += c
        if acc >= min_count:
            cuts.append(v); acc = 0
    if acc and cuts: cuts[-1] = vals[-1]
    return np.array(cuts if cuts else [vals[-1]])


def assign(f, cuts):
    return np.minimum(np.searchsorted(cuts, np.round(f, 9), side="left"), len(cuts) - 1)


def is_binary(y):
    return np.isin(np.unique(y), [0.0, 1.0]).all()


PRIOR = 2.0


def stratum_sd(y, overall_sd, binary=None, p_bar=None):
    """Within-stratum standard deviation that does not collapse to zero in sparse strata: for 0/1 values the
    proportion is shrunk toward the overall rate p_bar with prior strength PRIOR, (k + PRIOR p_bar)/(n + PRIOR);
    otherwise a floor of 10% of the overall s.d."""
    binary = is_binary(y) if binary is None and len(y) else binary
    if len(y) == 0: return overall_sd
    if binary:
        pb = 0.5 if p_bar is None else min(max(p_bar, 1e-3), 0.5)
        p = (y.sum() + PRIOR * pb) / (len(y) + PRIOR); return np.sqrt(p * (1 - p))
    return max(np.std(y, ddof=1) if len(y) > 1 else overall_sd, 0.1 * overall_sd)


def strat_plan(f_pilot, y_pilot, c0, c1, V, N, m_min=10):
    """Optimal (n, m1) for two-phase sampling for stratification with Neyman allocation:
    Var ~= A/n + B/m1 - S^2/N, A = (sum w_h S_h)^2, B = S^2 - sum w_h S_h^2; cost c0 n + c1 m1 (Theorem 3)."""
    cuts = strata_from_surrogate(f_pilot)
    h = assign(f_pilot, cuts)
    S2 = np.var(y_pilot, ddof=1)
    floor = 1.0 / len(y_pilot)
    w = np.array([(h == k).mean() for k in range(len(cuts))])
    pb = y_pilot.mean() if is_binary(y_pilot) else None
    sd = np.array([stratum_sd(y_pilot[h == k], np.sqrt(S2), p_bar=pb) for k in range(len(cuts))])
    A = np.sum(w * sd) ** 2; B = max(S2 - np.sum(w * sd ** 2), 0.0)
    T = V + S2 / N
    c1 = max(c1, 1e-9)
    root = np.sqrt(A * c0) + np.sqrt(B * c1)
    n, m1 = np.sqrt(A / c0) * root / T, np.sqrt(B / c1) * root / T
    if m1 > N:
        m1 = N; n = A / max(T - B / N, 1e-12)
    if m1 < n:  # nesting n <= m1 binds: both phases use the same sample, Var = (A+B)/n - S^2/N
        n = m1 = (A + B) / T
    n = min(max(n, m_min), N); m1 = min(max(m1, n), N)
    return dict(cuts=cuts, sd=sd, n=n, m1=m1, cost=c0 * n + max(c1, 0) * m1, A=A, B=B)


def estimate_strat(Y, f1, S1, cuts, sd, n, rng, N_rest, conf=0.95, p_bar=None):
    """Phase 1: S1 (uniform) labelled by the surrogate's labelers; phase 2: oracle sample in each stratum,
    Neyman allocation. Sparse strata are smoothed toward the current overall estimate. Returns estimate,
    half-width, number of oracle labels."""
    h = assign(f1, cuts); m1 = len(S1)
    w_hat = np.array([(h == k).mean() for k in range(len(cuts))])
    alloc = w_hat * sd; alloc = alloc / alloc.sum() * n if alloc.sum() > 0 else w_hat * n
    binary = is_binary(Y[S1[: min(len(S1), 2000)]])
    draws = []
    for k in range(len(cuts)):
        idx = S1[h == k]
        if len(idx) == 0: continue
        nk = int(min(len(idx), max(5, np.ceil(alloc[k]))))
        draws.append((k, len(idx), Y[idx[:nk]]))  # S1 is in random order: prefixes are nested across top-up rounds
    th = sum(w_hat[k] * yk.mean() for k, _, yk in draws)
    used = sum(len(yk) for _, _, yk in draws)
    v_within = 0.0
    for k, Mk, yk in draws:
        nk = len(yk)
        s2 = stratum_sd(yk, 0.0, binary, th) ** 2 if (binary or nk > 1) else 0.0
        v_within += w_hat[k] ** 2 * s2 / nk * (1 - nk / Mk)
    v_between = (1.0 / m1 - 1.0 / N_rest) * sum(w_hat[k] * (yk.mean() - th) ** 2 for k, _, yk in draws)
    return th, Z[conf] * np.sqrt(max(v_within + v_between, 0.0)), used
