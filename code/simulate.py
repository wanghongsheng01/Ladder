"""Offline evaluation of Ladder and baselines on pre-computed real LLM labels.

Every tuple has been labelled once by every labeler (label.py), so a query plan can be replayed many times
by sampling tuple ids; costs are the measured per-tuple API costs. Populations larger than the labelled sample
are built by resampling labelled tuples with replacement (each copy keeps its real labels and cost).
"""
import itertools, os
import numpy as np
import pandas as pd
import ladder as L
from tasks import TASKS, load, ROOT

LAB_DIR = os.path.join(ROOT, "results", "labels")
ALL_LABELERS = ["glm-flash", "glm-air", "ds-flash", "ds-flash-b10", "ds-flash-think", "kimi-k2.6", "ds-pro",
                "kimi-k3", "ds-pro-think"]
VENDOR = {"glm-flash": "GLM", "glm-air": "GLM", "ds-flash": "DeepSeek", "ds-flash-b10": "DeepSeek",
          "ds-flash-think": "DeepSeek", "ds-pro": "DeepSeek", "ds-pro-think": "DeepSeek", "kimi-k2.6": "Kimi",
          "kimi-k3": "Kimi", "code": "human code", "glm-flash-b10": "GLM", "glm-air-b10": "GLM", "kimi-k2.6-b10": "Kimi", "kimi-k3-b10": "Kimi"}

# query name -> (task, function gold-label-array -> Y)
QUERIES = {
    "imdb-pos": ("imdb", lambda g: (g == 1).astype(float)),
    "yelp-1star": ("yelp", lambda g: (g == 0).astype(float)),
    "yelp-avg": ("yelp", lambda g: g + 1.0),
    "agnews-biz": ("agnews", lambda g: (g == 2).astype(float)),
    "emotion-surprise": ("emotion", lambda g: (g == 5).astype(float)),
    "emotion-anger": ("emotion", lambda g: (g == 3).astype(float)),
    "hate": ("hate", lambda g: (g == 1).astype(float)),
    "offensive": ("offensive", lambda g: (g == 1).astype(float)),
    "amazon-avg": ("amazon", lambda g: g + 1.0),
    "amazon-pos": ("amazon", lambda g: (g >= 3).astype(float)),
}


def load_task(task, labelers):
    df = load(task)
    P, C = [], []
    for lab in labelers:
        d = pd.read_csv(os.path.join(LAB_DIR, f"{task}__{lab}.csv")).set_index("tid").loc[df.tid]
        P.append(d.pred.fillna(-1).astype(int).values)
        c = d.cost_usd.values.astype(float)
        C.append(np.where(np.isnan(c), np.nanmean(c), c))
    return df.gold.values, np.stack(P, 1), np.stack(C, 1)


def pilot_var(Yp):
    """Pilot variance with a floor p0(1-p0), p0 = 1/m_p, so that a pilot without a single positive tuple
    (rare predicates) does not plan an empty sample. Used by every method."""
    p0 = 1.0 / len(Yp)
    return max(np.var(Yp, ddof=1), p0 * (1 - p0))


class Setting:
    def __init__(self, query, labelers, c0=0.05, scale=1, oracle="human", seed=0):
        task, fy = QUERIES[query]
        self.task, self.labelers = task, list(labelers)
        need = self.labelers + ([oracle] if oracle != "human" else [])
        gold, P, C = load_task(task, need)
        if oracle != "human":  # LLM oracle: its labels define the truth, its measured cost is c0 per tuple
            k = need.index(oracle)
            keep = P[:, k] >= 0  # the query is defined over tuples the oracle can label (0.1-3% are unparsable)
            gold = P[keep, k]; c0 = float(C[keep, k].mean())
            P, C = P[keep][:, :len(self.labelers)], C[keep][:, :len(self.labelers)]
        self.C = len(TASKS[task]["labels"])
        rng = np.random.default_rng(seed)
        n = len(gold)
        pop = np.arange(n) if scale == 1 else rng.integers(0, n, size=int(scale * n))
        self.Y, self.P, self.Cost = fy(gold)[pop], P[pop], C[pop]
        self.N, self.c0 = len(pop), c0
        self.truth = self.Y.mean()


def plan_ladder(st, pilot, eps, conf, cand_labelers, bundles=True, max_len=3, m_min=10, strat=True, cv=True):
    """Choose the cheapest physical plan from the pilot: a control-variate ladder (Sec. III) or a two-phase
    stratified plan on one surrogate (Sec. IV-B). Returns a dict describing the plan."""
    Yp, Pp = st.Y[pilot], st.P[pilot]
    unit = {k: st.Cost[pilot, k].mean() for k in cand_labelers}
    members = [(k,) for k in cand_labelers]
    if bundles:
        members += list(itertools.combinations(cand_labelers, 2))
    cands = [(m, sum(unit[k] for k in m), L.cv_rho2(Pp, Yp, list(m), st.C)) for m in members]
    sig2 = pilot_var(Yp)
    N_rest = st.N - len(pilot)
    theta_p = max(Yp.mean(), 1.0 / len(pilot))
    V = (eps * theta_p / L.Z[conf]) ** 2
    u0 = V / sig2 + 1.0 / N_rest
    best = dict(kind="srs", m=np.array([min(N_rest, 1 / u0)]), cost=st.c0 * min(N_rest, 1 / u0))
    idx, score = L.best_chain(cands, unit, st.c0, max_len=max_len) if cv else ([], None)
    if idx:
        chain = [cands[i][0] for i in idx]
        m, cost = L.optimise(np.array(L.effective_costs(chain, unit)), np.array([cands[i][2] for i in idx]),
                             st.c0, sig2, V, N_rest, m_min)
        if m is not None and cost < best["cost"]:
            sur = [L.Surrogate(list(mem), st.C).fit(Pp, Yp) for mem in chain]
            alphas = []
            for s_ in sur:
                f = s_(Pp); v = np.var(f, ddof=1)
                alphas.append(np.cov(Yp, f)[0, 1] / v if v > 1e-12 else 0.0)
            best = dict(kind="cv", chain=chain, m=m, alphas=alphas, sur=sur, cost=cost)
    if strat:
        for mem, cst, r2 in cands:
            if r2 <= 0: continue
            sur = L.Surrogate(list(mem), st.C).fit(Pp, Yp)  # its values only label the cells used as strata
            plan = L.strat_plan(sur(Pp), Yp, st.c0, cst, V, N_rest, m_min)
            if plan["cost"] < best["cost"]:
                best = dict(kind="strat", chain=[mem], sur=sur, **plan)
    return best


def run_ladder(st, pilot, rest, eps, conf, rng, cand, **kw):
    plan = plan_ladder(st, pilot, eps, conf, cand, **kw)
    if plan["kind"] == "strat":
        m1 = int(min(len(rest), np.ceil(plan["m1"])))
        S1 = rng.permutation(rest)[:m1]
        f1 = plan["sur"](st.P[S1])
        th, hw, n_or = L.estimate_strat(st.Y, f1, S1, plan["cuts"], plan["sd"], plan["n"], rng, len(rest), conf,
                                        p_bar=st.Y[pilot].mean())
        mem = plan["chain"][0]
        cost = st.c0 * n_or + st.Cost[S1][:, list(mem)].sum()
        return th, hw, cost, plan["chain"], [n_or, m1]
    if plan["kind"] == "srs":
        return run_srs(st, pilot, rest, eps, conf, rng, cand)
    chain, m, alphas, sur = plan["chain"], plan["m"], plan["alphas"], plan["sur"]
    samples = L.nested_samples(rest, m, rng)
    th, hw = L.estimate_prefix(st.Y, st.P, sur, samples, alphas, len(rest), conf)
    cost = st.c0 * len(samples[0])
    for k in set().union(*chain):  # labeler k runs on the largest sample of a surrogate containing it
        j = max(i for i, mem in enumerate(chain) if k in mem) + 1
        cost += st.Cost[samples[j], k].sum()
    return th, hw, cost, chain, [len(s) for s in samples]


def run_ppi(st, pilot, rest, eps, conf, rng, cand):
    """PPI++ / regression estimator with one labeler applied to every remaining tuple (choice made on pilot to
    minimise total cost; Angelopoulos et al. 2023, Zrnic & Candes 2024)."""
    Yp, Pp = st.Y[pilot], st.P[pilot]
    sig2 = pilot_var(Yp); N_rest = len(rest)
    V = (eps * max(Yp.mean(), 1 / len(pilot)) / L.Z[conf]) ** 2
    best = None
    for k in cand:
        r2 = L.cv_rho2(Pp, Yp, [k], st.C)
        u0 = V / (sig2 * max(1 - r2, 1e-6)) + 1.0 / N_rest
        m0 = min(N_rest, 1 / u0)
        tot = st.c0 * m0 + st.Cost[pilot, k].mean() * N_rest
        if best is None or tot < best[0]: best = (tot, k, m0)
    _, k, m0 = best
    s = L.Surrogate([k], st.C).fit(Pp, Yp); f = s(Pp)
    a = np.cov(Yp, f)[0, 1] / max(np.var(f, ddof=1), 1e-12)
    samples = L.nested_samples(rest, np.array([m0, len(rest)]), rng)
    th, hw = L.estimate_prefix(st.Y, st.P, [s], samples, [a], N_rest, conf)
    return th, hw, st.c0 * len(samples[0]) + st.Cost[rest, k].sum(), [(k,)], [len(x) for x in samples]


def run_strat(st, pilot, rest, eps, conf, rng, cand):
    """Proxy-stratified sampling (ABae-style, Kang et al. VLDB 2021): one labeler on all tuples, strata = its
    predicted class, Neyman allocation of oracle labels estimated on the pilot."""
    Yp = st.Y[pilot]; N_rest = len(rest)
    V = (eps * max(Yp.mean(), 1 / len(pilot)) / L.Z[conf]) ** 2
    best = None
    for k in cand:
        strata = np.unique(st.P[rest, k])
        W = np.array([(st.P[rest, k] == h).mean() for h in strata])
        sd = []
        for h in strata:
            yh = Yp[st.P[pilot, k] == h]
            sd.append(np.std(yh, ddof=1) if len(yh) > 1 else np.sqrt(pilot_var(Yp)))
        sd = np.maximum(np.array(sd), 0.05 * np.sqrt(pilot_var(Yp)) + 1e-3)
        m = (np.sum(W * sd)) ** 2 / V
        tot = st.c0 * m + st.Cost[pilot, k].mean() * N_rest
        if best is None or tot < best[0]: best = (tot, k, strata, W, sd, m)
    _, k, strata, W, sd, m = best
    th = var = 0.0; n_or = 0
    for h, w, s in zip(strata, W, sd):
        ids = rest[st.P[rest, k] == h]
        mh = int(min(len(ids), max(2, np.ceil(m * w * s / np.sum(W * sd)))))
        smp = ids[:mh]; n_or += mh  # rest is in random order: a prefix is an SRSWOR, and top-up rounds extend it
        th += w * st.Y[smp].mean()
        var += w * w * np.var(st.Y[smp], ddof=1) / mh * (1 - mh / len(ids))
    return th, L.Z[conf] * np.sqrt(var), st.c0 * n_or + st.Cost[rest, k].sum(), [(k,)], [n_or]


def run_srs(st, pilot, rest, eps, conf, rng, cand):
    Yp = st.Y[pilot]; N_rest = len(rest)
    V = (eps * max(Yp.mean(), 1 / len(pilot)) / L.Z[conf]) ** 2
    m0 = min(N_rest, 1 / (V / pilot_var(Yp) + 1 / N_rest))
    s = rest[:max(2, int(np.ceil(m0)))]  # prefix of the random order, so top-up rounds extend the sample
    hw = L.Z[conf] * np.sqrt(np.var(st.Y[s], ddof=1) * (1 / len(s) - 1 / N_rest))
    return st.Y[s].mean(), hw, st.c0 * len(s), [], [len(s)]


METHODS = {"SRS": run_srs, "PPI++": run_ppi, "Strat": run_strat, "Ladder": run_ladder,
           "Ladder-nostrat": lambda *a: run_ladder(*a, strat=False),
           # stronger baselines: one proxy with an optimally sized (not full) sample, and proxy stratification with
           # the same sparse-stratum shrinkage and phase-1 sizing as Ladder's stratified plans
           "PPI++-sub": lambda *a: run_ladder(*a, bundles=False, max_len=1, strat=False),
           "Strat-shrink": lambda *a: run_ladder(*a, bundles=False, cv=False),
           "Ladder-nobundle": lambda *a: run_ladder(*a, bundles=False),
           "Ladder-len1": lambda *a: run_ladder(*a, max_len=1)}


def trial(st, method, eps=0.05, conf=0.95, pilot_n=200, seed=0, cand=None, max_rounds=8):
    """One run. Every method executes its plan, and if the reported half-width exceeds eps times its own estimate it
    tops up: the plan is re-sized for a proportionally tighter target and extended (same random order, so the new
    samples extend the old ones), up to max_rounds. The cost of the final round is charged."""
    cand = list(range(len(st.labelers))) if cand is None else cand
    eps_eff = eps
    for rnd in range(max_rounds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(st.N); pilot, rest = perm[:pilot_n], perm[pilot_n:]
        th, hw, cost, chain, sizes = METHODS[method](st, pilot, rest, eps_eff, conf, rng, cand)
        Nr_ = len(rest); est_ = (st.Y[pilot].sum() + Nr_ * th) / st.N; hw_ = hw * Nr_ / st.N
        if hw_ <= eps * abs(est_) + 1e-12 or (sizes and sizes[0] >= Nr_):  # met, or the oracle saw everything
            break
        eps_eff *= min(0.9, max(0.5, 0.9 * eps * abs(est_) / max(hw_, 1e-12)))  # tighten, at most halve per round
    pilot_cost = st.c0 * pilot_n + (0.0 if method == "SRS" else st.Cost[pilot][:, cand].sum())
    Nr = len(rest)
    est = (st.Y[pilot].sum() + Nr * th) / st.N
    hw = hw * Nr / st.N
    return dict(method=method, est=est, hw=hw, truth=st.truth, covered=abs(est - st.truth) <= hw,
                rel_err=abs(est - st.truth) / st.truth, rel_hw=hw / st.truth, cost=cost + pilot_cost,
                cost_main=cost, rounds=rnd + 1, met=hw <= eps * abs(est) + 1e-12, chain="|".join("+".join(st.labelers[k] for k in m) for m in chain),
                sizes=sizes)


# ---------------------------------------------------------------- GROUP BY COUNT
GROUPBY = {"agnews-topic": "agnews", "emotion-all": "emotion", "yelp-stars": "yelp", "amazon-stars": "amazon"}


class GroupSetting(Setting):
    def __init__(self, query, labelers, c0=0.05, scale=1, oracle="human", seed=0):
        task = GROUPBY[query]
        QUERIES["_tmp"] = (task, lambda g: g.astype(float))
        super().__init__("_tmp", labelers, c0, scale, oracle, seed)
        self.G = len(TASKS[task]["labels"])
        self.Yg = np.stack([(self.Y == g).astype(float) for g in range(self.G)])  # (G, N)
        self.truth_g = self.Yg.mean(1)


def trial_groupby(st, method, eps=0.1, conf=0.95, pilot_n=300, seed=0, max_rounds=8, **kw):
    """GROUP BY with the same top-up rule as trial(): while the largest reported relative half-width exceeds eps,
    re-plan for a proportionally tighter target with the same random order (the samples extend), up to max_rounds.
    The cost of the final round is charged."""
    eps_eff = eps
    for rnd in range(max_rounds):
        r = run_groupby(st, method, eps_eff, conf, pilot_n, seed, **kw)
        ratio = r["max_rel_hw_est"]
        if ratio <= eps + 1e-12 or r["sizes"][0] >= st.N - pilot_n:  # met, or the oracle saw everything
            break
        eps_eff *= min(0.9, max(0.5, 0.9 * eps / ratio))  # tighten, at most halve per round
    r.update(rounds=rnd + 1, met=ratio <= eps + 1e-12)
    return r


def run_groupby(st, method, eps=0.1, conf=0.95, pilot_n=300, seed=0, bundles=True, max_len=2, m_min=10):
    """Shared plan for all groups; per-group ladder estimators with group-specific coefficients."""
    rng = np.random.default_rng(seed)
    cand = list(range(len(st.labelers)))
    perm = rng.permutation(st.N); pilot, rest = perm[:pilot_n], perm[pilot_n:]
    N_rest = len(rest); G = st.G
    Yp = st.Yg[:, pilot]; Pp = st.P[pilot]
    sig2 = np.array([pilot_var(y) for y in Yp])
    th_p = np.maximum(Yp.mean(1), 1.0 / pilot_n)
    V = (eps * th_p / L.Z[conf]) ** 2
    pilot_cost = st.c0 * pilot_n + (0 if method == "SRS" else st.Cost[pilot].sum())
    chain, m = [], None
    if method == "SRS":
        u0 = np.min(V / sig2) + 1.0 / N_rest; m = np.array([min(N_rest, 1 / u0)])
    else:
        unit = {k: st.Cost[pilot, k].mean() for k in cand}
        members = [(k,) for k in cand] + (list(itertools.combinations(cand, 2)) if bundles else [])
        if method == "PPI++":
            members = [(k,) for k in cand]
        r2 = {mem: np.array([L.cv_rho2(Pp, Yp[g], list(mem), st.C) for g in range(G)]) for mem in members}
        best = (np.inf, None, None)
        if method == "PPI++":  # one labeler on all remaining tuples
            for mem in members:
                u0 = np.min(V / (sig2 * np.maximum(1 - r2[mem], 1e-6))) + 1.0 / N_rest
                tot = st.c0 / u0 + unit[mem[0]] * N_rest
                if tot < best[0]: best = (tot, [mem], np.array([1 / u0, N_rest]))
        else:
            top = sorted(members, key=lambda mm: -r2[mm].mean())[:12]
            chains = [[a] for a in top] + ([[a, b] for a in top for b in top if r2[a].mean() > r2[b].mean()] if max_len >= 2 else [])
            for ch in chains:
                ceff = L.effective_costs(ch, unit)
                if min(ceff) <= 0: continue
                R = np.stack([r2[mm] for mm in ch], 1)
                mm_, cost = L.optimise_multi(np.array(ceff), R, st.c0, sig2, V, N_rest, m_min)
                if cost < best[0]: best = (cost, ch, mm_)
        if best[1] is None:
            u0 = np.min(V / sig2) + 1.0 / N_rest; m = np.array([min(N_rest, 1 / u0)])
        else:
            chain, m = best[1], best[2]
    samples = L.nested_samples(rest, m, rng)
    cost = st.c0 * len(samples[0])
    for k in (set().union(*chain) if chain else set()):
        j = max(i for i, mem in enumerate(chain) if k in mem) + 1
        cost += st.Cost[samples[j], k].sum()
    out = []
    for g in range(G):
        sur = [L.Surrogate(list(mem), st.C).fit(Pp, Yp[g]) for mem in chain]
        al = []
        for s in sur:
            f = s(Pp); v = np.var(f, ddof=1); al.append(np.cov(Yp[g], f)[0, 1] / v if v > 1e-12 else 0.0)
        th, hw = L.estimate_prefix(st.Yg[g], st.P, sur, samples, al, N_rest, conf)
        est = (Yp[g].sum() + N_rest * th) / st.N; hw = hw * N_rest / st.N
        out.append((est, hw, abs(est - st.truth_g[g]) <= hw + 1e-12))
    est, hw, cov = map(np.array, zip(*out))
    return dict(method=method, cost=cost + pilot_cost, all_covered=cov.all(), cov_rate=cov.mean(),
                max_rel_hw=np.max(hw / st.truth_g), max_rel_err=np.max(np.abs(est - st.truth_g) / st.truth_g),
                max_rel_hw_est=np.max(hw / np.maximum(np.abs(est), 1e-12)),
                chain="|".join("+".join(st.labelers[k] for k in mm) for mm in chain), sizes=[len(s) for s in samples])
